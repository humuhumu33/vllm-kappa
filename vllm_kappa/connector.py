"""L1 — the κ-witness KV connector.

Scheduler-side: on request_finished (called exactly once per request, before
its blocks are freed), the full token stream + sampling fingerprint are
queued off-path; a daemon thread builds the block-witness chain and seal
(witness.build_chain) and puts them in the κ-store. Serving never waits.

Worker-side (VLLM_KAPPA_WITNESS_KV=1, experimental until the WSL integration
gate): prefill KV bytes are digested per block — the extraction mirrors
vLLM's in-tree ExampleConnector — and stored as `kvd` records indexed by
block key, giving the seal's blocks byte-level KV evidence.

Wire-up (no vLLM changes; out-of-tree registration like LMCache's):

    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory)
    KVConnectorFactory.register_connector(
        "KappaConnector", "vllm_kappa.connector", "KappaConnector")

then --kv-transfer-config '{"kv_connector": "KappaConnector",
                            "kv_role": "kv_both"}'.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import PINNED_VLLM_SHA, assert_seams
from .addressing import chain_block_keys
from .fabric import KappaStore, store_from_env
from .witness import EngineFingerprint, build_chain, request_kappa

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

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
    """Off-path chain builder shared by scheduler- and worker-side roles."""

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
                label = self.store.put(seal)
                self.counters["sealed"] += 1
                logger.debug(
                    "sealed request %s (%d blocks) -> %s",
                    request_kappa(seal),
                    len(records),
                    label,
                )
            except Exception:
                logger.exception("witness build failed (request dropped)")
                self.counters["dropped"] += 1

    def close(self) -> None:
        self._q.put(None)
        self._t.join(timeout=5)


_Base, _Meta, _Role = (None, None, None)


def _bases():
    global _Base, _Meta, _Role
    if _Base is None:
        _Base, _Meta, _Role = _import_base()
    return _Base, _Meta, _Role


def KappaConnector(*args, **kwargs):  # noqa: N802 - class path used by vLLM
    """Factory: builds the real class on first use (keeps module importable
    without torch/vLLM for the verifier and unit tests)."""
    return _build_class()(*args, **kwargs)


_cls_cache = None


def _build_class():
    global _cls_cache
    if _cls_cache is not None:
        return _cls_cache

    import torch  # noqa: F401

    Base, Meta, Role = _bases()

    @dataclass
    class _StoreReq:
        token_ids: list[int]
        block_ids: list[int]

    @dataclass
    class KappaConnectorMetadata(Meta):
        stores: list[_StoreReq] = field(default_factory=list)

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
            self._writer = _WitnessWriter(self._engine, self._store)
            self._witness_kv = os.getenv("VLLM_KAPPA_WITNESS_KV", "0") == "1"
            self._kv_hashers: dict[int, Any] = {}
            logger.info(
                "kappa-connector up (role=%s, kv-witness=%s, batch_invariant=%s)",
                role,
                self._witness_kv,
                self._engine.batch_invariant,
            )
            if not self._engine.batch_invariant:
                logger.warning(
                    "VLLM_BATCH_INVARIANT is off: witnesses record "
                    "per-deployment bytes, not canonical bytes (PROMPT.md L0)."
                )

        # ---------- scheduler side ----------
        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 0, False  # L3 (κ-KV pull) lands in Phase 3

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            return

        def build_connector_meta(self, scheduler_output):
            meta = KappaConnectorMetadata()
            if self._witness_kv:
                for new_req in scheduler_output.scheduled_new_reqs:
                    meta.stores.append(
                        _StoreReq(
                            token_ids=list(new_req.prompt_token_ids or []),
                            block_ids=list(new_req.block_ids[0]),
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

        # ---------- worker side ----------
        def register_kv_caches(self, kv_caches):
            self._layer_names = list(kv_caches)

        def start_load_kv(self, forward_context, **kwargs):
            return

        def wait_for_layer_load(self, layer_name):
            return

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            if not self._witness_kv:
                return
            import blake3
            import torch as _torch

            meta = self._get_connector_metadata()
            if not isinstance(meta, KappaConnectorMetadata):
                return
            bs = self._block_size
            for req in meta.stores:
                n_full = (len(req.token_ids) // bs) * bs
                if n_full == 0:
                    continue
                # mirror ExampleConnector's layout assumption:
                # (num_pages, 2, page_size, ...) non-MLA paged buffer
                blocks = req.block_ids[: n_full // bs]
                for j, block_id in enumerate(blocks):
                    data = (
                        kv_layer[block_id].detach().to("cpu", non_blocking=False)
                    )
                    h = self._kv_hashers.setdefault(
                        (id(req), j), blake3.blake3()
                    )
                    h.update(_torch.flatten(data).contiguous().numpy().tobytes())

        def wait_for_save(self):
            if not self._witness_kv or not self._kv_hashers:
                return
            import cbor2

            meta = self._get_connector_metadata()
            if not isinstance(meta, KappaConnectorMetadata):
                return
            for req in meta.stores:
                bs = self._block_size
                keys = chain_block_keys(
                    req.token_ids, bs, seed=self._engine.none_hash_seed
                )
                for j, bk in enumerate(keys):
                    h = self._kv_hashers.pop((id(req), j), None)
                    if h is None:
                        continue
                    self._store.put(
                        cbor2.dumps(
                            {"v": 1, "t": "kvd", "bk": bk, "kv": h.digest()},
                            canonical=True,
                        )
                    )
            self._kv_hashers.clear()

        def shutdown(self):
            self._writer.close()
            self._store.close()

    _cls_cache = _KappaConnector
    return _cls_cache
