"""L1 + L3 — the κ-witness / κ-KV connector.

Scheduler-side: on request_finished (called exactly once per request, before
its blocks are freed), the full token stream + sampling fingerprint are
queued off-path; a daemon thread builds the block-witness chain and seal
(witness.build_chain) and puts them in the κ-store. Serving never waits.

Worker-side, opt-in by env:
  VLLM_KAPPA_WITNESS_KV=1 — per-block BLAKE3 digests of prefill KV bytes
      (`kvd` records) giving the seal byte-level KV evidence.
  VLLM_KAPPA_KV=1 — L3: full prefill KV payloads stored per BLOCK, keyed by
      the block-κ through the local KeyedIndex. A later instance (or boot)
      that sees the same prefix pulls those blocks instead of prefilling:
      get_num_new_matched_tokens reports the contiguous κ-hit, start_load_kv
      fetches (verify-on-read — a tampered payload fails its label) and
      injects into the paged buffer; fetch failures are reported through
      get_block_ids_with_load_errors so vLLM recomputes those blocks.
      Implies WITNESS_KV.

Layout assumption (mirrors the in-tree ExampleConnector, non-MLA):
paged buffer per layer is [num_pages, 2, page_size, ...]. Anything else
disables the KV paths with a counter rather than corrupting state.

Wire-up (no vLLM changes):
    --kv-transfer-config '{"kv_connector": "KappaConnector",
        "kv_connector_module_path": "vllm_kappa.connector",
        "kv_role": "kv_both"}'
"""

from __future__ import annotations

import io
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import PINNED_VLLM_SHA, assert_seams
from .addressing import chain_block_keys
from .fabric import IntegrityError, KappaStore, KeyedIndex, store_from_env
from .witness import EngineFingerprint, build_chain, request_kappa

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger("vllm_kappa.connector")

_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "max_tokens",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
)


def sampling_fingerprint(sampling_params: Any) -> dict:
    """T6: sampling is part of witness identity. Stable, canonical subset."""
    return {
        f: getattr(sampling_params, f, None)
        for f in _SAMPLING_FIELDS
        if getattr(sampling_params, f, None) is not None
    }


@dataclass
class _WitnessJob:
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    sampling: dict


@dataclass
class _StoreReq:
    token_ids: list[int]
    block_ids: list[int]
    # blocks in [skip_start, skip_end) were externally claimed (loaded or
    # refused), NOT computed here — storing them would snapshot whatever the
    # paged buffer holds (garbage before recompute) under a valid κ,
    # replacing detectable corruption with undetectable poison. Never store
    # what you did not compute.
    skip_start: int = 0
    skip_end: int = 0


@dataclass
class _LoadReq:
    token_ids: list[int]
    block_ids: list[int]
    start_block: int  # first block index to inject (aligned)
    end_block: int  # one past the last block index to inject


def _align_down(n: int, block: int) -> int:
    return (n // block) * block


def _import_base():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )

    return KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole


def _engine_fingerprint(vllm_config: "VllmConfig") -> EngineFingerprint:
    import vllm
    import vllm.envs as envs

    try:
        from vllm.v1.core.kv_cache_utils import get_none_hash_seed

        seed = get_none_hash_seed()
    except ImportError:
        from .addressing import NONE_HASH_SEED

        seed = NONE_HASH_SEED

    return EngineFingerprint(
        vllm_sha=getattr(vllm, "__commit__", None) or PINNED_VLLM_SHA,
        model_root=vllm_config.model_config.model,
        dtype=str(vllm_config.model_config.dtype),
        tp_size=vllm_config.parallel_config.tensor_parallel_size,
        block_size=vllm_config.cache_config.block_size,
        batch_invariant=bool(envs.VLLM_BATCH_INVARIANT),
        none_hash_seed=seed,
    )


class _WitnessWriter:
    """Off-path chain builder (scheduler side)."""

    def __init__(self, engine: EngineFingerprint, store: KappaStore):
        self.engine = engine
        self.store = store
        self.counters = {"sealed": 0, "dropped": 0}
        self._q: queue.Queue[_WitnessJob | None] = queue.Queue(maxsize=2048)
        self._t = threading.Thread(
            target=self._drain, name="kappa-witness", daemon=True
        )
        self._t.start()

    def submit(self, job: _WitnessJob) -> None:
        try:
            self._q.put_nowait(job)
        except queue.Full:
            self.counters["dropped"] += 1

    def _drain(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            try:
                records, seal = build_chain(
                    self.engine,
                    job.sampling,
                    job.prompt_token_ids,
                    job.output_token_ids,
                )
                for rec in records:
                    self.store.put(rec)
                self.store.put(seal)
                self.counters["sealed"] += 1
                logger.debug("sealed %s", request_kappa(seal))
            except Exception:
                logger.exception("witness build failed (request dropped)")
                self.counters["dropped"] += 1

    def close(self) -> None:
        self._q.put(None)
        self._t.join(timeout=5)


_lazy: dict[str, type] = {}


def __getattr__(name):  # PEP 562: vLLM resolves these CLASSES by name — for
    # classmethod calls on the connector and for pickling the metadata
    # between the scheduler and worker processes — so the lazy build must
    # yield real module-addressable classes. The module itself stays
    # importable without torch for the verifier and unit tests.
    if name in ("KappaConnector", "KappaConnectorMetadata"):
        _build_class()
        return _lazy[name]
    raise AttributeError(name)


def _build_class():
    if "KappaConnector" in _lazy:
        return _lazy["KappaConnector"]

    import torch

    Base, Meta, Role = _import_base()

    @dataclass
    class KappaConnectorMetadata(Meta):
        stores: list[_StoreReq] = field(default_factory=list)
        loads: list[_LoadReq] = field(default_factory=list)

    KappaConnectorMetadata.__module__ = __name__
    KappaConnectorMetadata.__qualname__ = "KappaConnectorMetadata"

    class _KappaConnector(Base):
        def __init__(self, vllm_config, role, kv_cache_config=None, **kw):
            assert_seams()
            try:
                super().__init__(
                    vllm_config=vllm_config,
                    role=role,
                    kv_cache_config=kv_cache_config,
                    **kw,
                )
            except TypeError:  # T4: ctor kwargs move between versions
                super().__init__(vllm_config=vllm_config, role=role)
            self._block_size = vllm_config.cache_config.block_size
            self._engine = _engine_fingerprint(vllm_config)
            self._store = store_from_env()
            self._index = KeyedIndex(self._store.local.root)
            self._writer = _WitnessWriter(self._engine, self._store)
            self._kv_payloads = os.getenv("VLLM_KAPPA_KV", "0") == "1"
            self._witness_kv = (
                self._kv_payloads
                or os.getenv("VLLM_KAPPA_WITNESS_KV", "0") == "1"
            )
            self._requests_need_load: dict[str, tuple[int, int]] = {}
            # worker-side scratch: (id(req-meta), block-idx) -> {layer: rows}
            self._pending: dict[tuple[int, int], dict[str, Any]] = {}
            self._load_errors: set[int] = set()
            self.counters = {
                "kv_blocks_stored": 0,
                "kv_blocks_injected": 0,
                "kv_load_errors": 0,
                "kv_layout_skips": 0,
                "kappa_hits_reported": 0,
            }
            logger.info(
                "kappa-connector up (role=%s, kv=%s, witness_kv=%s, bi=%s)",
                role,
                self._kv_payloads,
                self._witness_kv,
                self._engine.batch_invariant,
            )
            if not self._engine.batch_invariant:
                logger.warning(
                    "VLLM_BATCH_INVARIANT off: witnesses record per-regime "
                    "bytes, not canonical bytes (PROMPT.md L0/G0b)."
                )

        # ================= scheduler side =================
        def _prompt_block_keys(self, request) -> list[bytes]:
            tokens = list(request.prompt_token_ids or [])
            # the final token must be computed to produce logits, so only
            # blocks fully inside len-1 are eligible (ExampleConnector rule)
            usable = _align_down(len(tokens) - 1, self._block_size)
            return chain_block_keys(
                tokens[:usable], self._block_size, seed=self._engine.none_hash_seed
            )

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            if not self._kv_payloads:
                return 0, False
            keys = self._prompt_block_keys(request)
            start = num_computed_tokens // self._block_size
            hit = start
            while hit < len(keys) and self._index.get(keys[hit]) is not None:
                hit += 1
            n_new = (hit - start) * self._block_size
            if n_new > 0:
                self._requests_need_load[request.request_id] = (start, hit)
                self.counters["kappa_hits_reported"] += 1
                logger.info(
                    "κ-KV hit: request %s blocks [%d,%d) — %d tokens skipped",
                    request.request_id,
                    start,
                    hit,
                    n_new,
                )
            return n_new, False

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            if num_external_tokens == 0:
                self._requests_need_load.pop(request.request_id, None)

        def build_connector_meta(self, scheduler_output):
            meta = KappaConnectorMetadata()
            if not self._witness_kv:
                return meta
            for new_req in scheduler_output.scheduled_new_reqs:
                token_ids = list(new_req.prompt_token_ids or [])
                block_ids = list(new_req.block_ids[0])
                window = self._requests_need_load.pop(new_req.req_id, None)
                if window is not None:
                    meta.loads.append(
                        _LoadReq(
                            token_ids=token_ids,
                            block_ids=block_ids,
                            start_block=window[0],
                            end_block=window[1],
                        )
                    )
                if self._kv_payloads or self._witness_kv:
                    meta.stores.append(
                        _StoreReq(
                            token_ids=token_ids,
                            block_ids=block_ids,
                            skip_start=window[0] if window else 0,
                            skip_end=window[1] if window else 0,
                        )
                    )
            return meta

        def request_finished(self, request, block_ids):
            sp = getattr(request, "sampling_params", None)
            self._writer.submit(
                _WitnessJob(
                    prompt_token_ids=list(request.prompt_token_ids or []),
                    output_token_ids=list(request.output_token_ids or []),
                    sampling=sampling_fingerprint(sp) if sp else {},
                )
            )
            return False, None

        # ================= worker side =================
        def register_kv_caches(self, kv_caches):
            self._layer_names = list(kv_caches)

        def _layer_rows(self, kv_layer, slot_mapping):
            """Extract per-token KV rows; None if the layout is foreign."""
            if kv_layer.dim() < 3 or kv_layer.shape[1] != 2:
                self.counters["kv_layout_skips"] += 1
                return None
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            return kv_layer[block_idxs, :, offsets]

        def start_load_kv(self, forward_context, **kwargs):
            meta = self._get_connector_metadata()
            if not isinstance(meta, KappaConnectorMetadata) or not meta.loads:
                return
            bs = self._block_size
            for req in meta.loads:
                keys = chain_block_keys(
                    req.token_ids[
                        : _align_down(len(req.token_ids) - 1, bs)
                    ],
                    bs,
                    seed=self._engine.none_hash_seed,
                )
                for j in range(req.start_block, req.end_block):
                    dest_block = req.block_ids[j]
                    payload = None
                    label = self._index.get(keys[j]) if j < len(keys) else None
                    if label is not None:
                        try:
                            raw = self._store.get(label)
                        except IntegrityError:
                            raw = None
                        if raw is not None:
                            try:
                                payload = torch.load(
                                    io.BytesIO(raw), weights_only=True
                                )
                            except Exception:
                                payload = None
                    if payload is None:
                        self._load_errors.add(dest_block)
                        self.counters["kv_load_errors"] += 1
                        if j < len(keys):
                            # drop the mapping so the rescheduled request
                            # misses and prefills instead of re-claiming the
                            # same refused payload forever
                            self._index.drop(keys[j])
                        logger.warning(
                            "κ-KV load refused: prompt block %d (dest block "
                            "%d) — index entry dropped, reporting for recompute",
                            j,
                            dest_block,
                        )
                        continue
                    slot = torch.arange(
                        dest_block * bs, dest_block * bs + bs
                    )
                    block_idxs = slot // bs
                    offsets = slot % bs
                    ok = True
                    for name in self._layer_names:
                        layer = forward_context.no_compile_layers.get(name)
                        kv = getattr(layer, "kv_cache", None) if layer else None
                        if isinstance(kv, (list, tuple)):
                            kv = kv[getattr(forward_context, "virtual_engine", 0)]
                        rows = payload.get(name)
                        if kv is None or rows is None or kv.shape[1] != 2:
                            ok = False
                            break
                        kv[block_idxs, :, offsets] = rows.to(kv.device)
                    if ok:
                        self.counters["kv_blocks_injected"] += 1
                    else:
                        self._load_errors.add(dest_block)
                        self.counters["kv_load_errors"] += 1

        def get_block_ids_with_load_errors(self):
            errs, self._load_errors = self._load_errors, set()
            if errs:
                logger.warning("κ-KV reporting %d invalid blocks: %s", len(errs), errs)
            return errs

        def wait_for_layer_load(self, layer_name):
            return

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            if not self._witness_kv:
                return
            meta = self._get_connector_metadata()
            if not isinstance(meta, KappaConnectorMetadata):
                return
            bs = self._block_size
            for req in meta.stores:
                n_full = _align_down(len(req.token_ids) - 1, bs)
                if n_full == 0:
                    continue
                slot = torch.tensor(
                    [
                        req.block_ids[i // bs] * bs + (i % bs)
                        for i in range(n_full)
                    ]
                )
                rows = self._layer_rows(kv_layer, slot)
                if rows is None:
                    return
                rows = rows.detach().to("cpu")
                for j in range(n_full // bs):
                    self._pending.setdefault((id(req), j), {})[layer_name] = (
                        rows[j * bs : (j + 1) * bs].clone()
                    )

        def wait_for_save(self):
            if not self._witness_kv or not self._pending:
                return
            import blake3
            import cbor2

            meta = self._get_connector_metadata()
            if not isinstance(meta, KappaConnectorMetadata):
                return
            bs = self._block_size
            for req in meta.stores:
                n_full = _align_down(len(req.token_ids) - 1, bs)
                keys = chain_block_keys(
                    req.token_ids[:n_full],
                    bs,
                    seed=self._engine.none_hash_seed,
                )
                for j, bk in enumerate(keys):
                    layers = self._pending.pop((id(req), j), None)
                    if layers is None:
                        continue
                    if req.skip_start <= j < req.skip_end:
                        continue  # externally claimed — not ours to store
                    if self._index.get(bk) is not None:
                        continue  # compute-once: block already stored
                    buf = io.BytesIO()
                    torch.save(layers, buf)
                    raw = buf.getvalue()
                    label = self._store.put(raw)
                    self._index.put(bk, label)
                    self._store.put(
                        cbor2.dumps(
                            {
                                "v": 1,
                                "t": "kvd",
                                "bk": bk,
                                "kv": blake3.blake3(raw).digest(),
                            },
                            canonical=True,
                        )
                    )
                    self.counters["kv_blocks_stored"] += 1
                logger.warning(
                    "κ-KV store pass: %d/%d full blocks now indexed for "
                    "request of %d tokens",
                    len(keys),
                    n_full // bs,
                    len(req.token_ids),
                )
            self._pending.clear()

        def shutdown(self):
            self._writer.close()
            self._store.close()

    _KappaConnector.__module__ = __name__
    _KappaConnector.__qualname__ = "KappaConnector"
    _lazy["KappaConnector"] = _KappaConnector
    _lazy["KappaConnectorMetadata"] = KappaConnectorMetadata
    return _KappaConnector
