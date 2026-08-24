"""L2 — the κ-drafter.

KappaProposer subclasses vLLM's own SuffixDecodingProposer (which wraps the
Arctic Inference suffix tree), adding exactly two behaviors:

1. persistence: when a request finishes, its (pattern, accepted-response)
   pair is sealed to the κ-store — accepted tokens were sampled by the target
   model, i.e. verified by construction;
2. warm start: on boot, sealed draft records are replayed into the suffix
   tree as synthetic cached requests, so the first request after a restart
   (or on a second instance sharing the store) drafts immediately.

The hot path is untouched: propose() delegates to the in-memory tree; the
store is only read at warm start and written off-path. A poisoned store can
only lower acceptance rates — rejection sampling in the target model means
output tokens cannot be corrupted (gate G2b asserts this).

Wire-up (no vLLM changes):
  --speculative-config '{"method": "custom",
      "model": "vllm_kappa.drafter.KappaProposer",
      "num_speculative_tokens": 8}'
"""

from __future__ import annotations

import logging
import os
import queue
import threading

import cbor2

from . import assert_seams
from .fabric import KappaStore, store_from_env

logger = logging.getLogger("vllm_kappa.drafter")

DRAFT_SCHEMA = 1
MIN_SEAL_TOKENS = 4  # don't persist trivial continuations


def encode_draft(pattern: list[int], response: list[int]) -> bytes:
    return cbor2.dumps(
        {"v": DRAFT_SCHEMA, "t": "draft", "pat": pattern, "resp": response},
        canonical=True,
    )


def decode_draft(data: bytes) -> tuple[list[int], list[int]] | None:
    try:
        rec = cbor2.loads(data)
    except Exception:
        return None
    if isinstance(rec, dict) and rec.get("v") == DRAFT_SCHEMA and rec.get("t") == "draft":
        return list(rec["pat"]), list(rec["resp"])
    return None


# SuffixDecodingProposer imports torch and arctic_inference at module import;
# subclass at runtime so this module stays importable in verifier/test
# environments without them.
def _build_class():
    from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

    class _KappaProposer(SuffixDecodingProposer):
        def __init__(self, vllm_config):
            assert_seams()
            super().__init__(vllm_config)
            self.store: KappaStore = store_from_env()
            self.counters = {
                "warm_records": 0,
                "sealed_records": 0,
                "tracked_requests": 0,
            }
            # req_id -> (pattern, [accepted token ids so far])
            self._tracked: dict[str, tuple[list[int], list[int]]] = {}
            self._seal_q: queue.Queue[bytes | None] = queue.Queue(maxsize=1024)
            self._sealer = threading.Thread(
                target=self._drain_seals, name="kappa-draft-seal", daemon=True
            )
            self._sealer.start()
            self._warm(int(os.getenv("VLLM_KAPPA_WARM_MAX", "5000")))
            # continuous batching never calls propose again after the last
            # step, so requests that finish together would otherwise never
            # seal; flush synchronously at interpreter exit.
            import atexit

            atexit.register(self._seal_all)

        def _seal_all(self) -> None:
            for rid in list(self._tracked):
                pattern, response = self._tracked.pop(rid, (None, None))
                if pattern and response and len(response) >= MIN_SEAL_TOKENS:
                    self.store.put(encode_draft(pattern, response))
                    self.counters["sealed_records"] += 1

        # -- persistence (off-path) ---------------------------------------
        def _drain_seals(self):
            while True:
                data = self._seal_q.get()
                if data is None:
                    return
                self.store.put(data)
                self.counters["sealed_records"] += 1

        def _seal(self, req_id: str) -> None:
            pattern, response = self._tracked.pop(req_id, (None, None))
            if not pattern or not response or len(response) < MIN_SEAL_TOKENS:
                return
            try:
                self._seal_q.put_nowait(encode_draft(pattern, response))
            except queue.Full:
                pass  # never block serving on persistence backlog

        # -- warm start ----------------------------------------------------
        def _warm(self, limit: int) -> None:
            n = 0
            for label in self.store.local.labels():
                if n >= limit:
                    break
                data = self.store.get(label)
                if data is None:
                    continue
                decoded = decode_draft(data)
                if decoded is None:
                    continue  # not a draft record (witness etc.)
                pattern, response = decoded
                synth = f"kappa-warm-{n}"
                try:
                    self.suffix_cache.start_request(synth, pattern)
                    self.suffix_cache.add_active_response(synth, response)
                    self.suffix_cache.stop_request(synth)
                    n += 1
                except Exception as e:  # warm start must never block boot
                    logger.debug("warm replay failed for %s: %s", label, e)
            self.counters["warm_records"] = n
            if n:
                logger.info("kappa-drafter: warmed suffix tree with %d records", n)

        # -- the hot path ---------------------------------------------------
        # Two propose contracts exist (T4):
        #   pinned era:  propose(num_spec, input_batch, sampled_token_ids, ...)
        #   nightly:     propose(sampled_token_ids, num_tokens_no_spec,
        #                        token_ids_cpu, slot_mappings=...)
        # The nightly contract carries no req_ids, so rows are identified by
        # a stable content key (blake3 of the first tokens of the context).
        # Two requests with an identical 32-token prefix share an identity;
        # that only merges their draft bookkeeping, never their outputs.
        def propose(self, *args, slot_mappings=None, **_kw):
            if args and isinstance(args[0], int):
                num_spec, input_batch, sampled = args[0], args[1], args[2]
                num_tokens = input_batch.num_tokens_no_spec
                token_mat = input_batch.token_ids_cpu
                req_ids = list(input_batch.req_ids)
            else:
                sampled, num_tokens, token_mat = args[0], args[1], args[2]
                req_ids = None
                num_spec = self.num_speculative_tokens
            return self._propose_rows(sampled, num_tokens, token_mat, req_ids, num_spec)

        def _row_key(self, ctx):
            import blake3 as _b3

            head = bytes(str(ctx[: min(len(ctx), 32)]), "utf-8")
            return "k" + _b3.blake3(head).hexdigest()[:16]

        def _propose_rows(self, sampled, num_tokens, token_mat, req_ids, num_spec):
            drafts: list[list[int]] = []
            seen: set[str] = set()
            for i, sampled_ids in enumerate(sampled):
                if sampled_ids is None or len(sampled_ids) == 0:
                    drafts.append([])
                    continue
                n = int(num_tokens[i])
                if n >= self.max_model_len - 1:
                    drafts.append([])
                    continue
                ctx = [int(t) for t in token_mat[i, :n]]
                rid = req_ids[i] if req_ids is not None else self._row_key(ctx)
                seen.add(rid)

                if rid not in self.suffix_cache.active_requests:
                    if rid in self.suffix_cache.cached_requests:
                        self.suffix_cache.evict_cached_response(rid)
                    self.suffix_cache.start_request(rid, ctx)
                    start = max(0, n - len(sampled_ids) - self.max_tree_depth)
                    self._tracked[rid] = (
                        ctx[start : n - len(sampled_ids)],
                        [int(t) for t in sampled_ids],
                    )
                    self.counters["tracked_requests"] += 1
                else:
                    self.suffix_cache.add_active_response(
                        rid, [int(t) for t in sampled_ids]
                    )
                    if rid in self._tracked:
                        self._tracked[rid][1].extend(int(t) for t in sampled_ids)

                pattern = ctx[max(0, n - self.max_tree_depth) :]
                draft = self.suffix_cache.speculate(
                    rid,
                    pattern,
                    max_spec_tokens=min(num_spec, self.max_model_len - n - 1),
                    max_spec_factor=self.max_spec_factor,
                    min_token_prob=self.min_token_prob,
                )
                drafts.append(list(draft.token_ids))

            for rid in list(self.suffix_cache.active_requests - seen):
                self._seal(rid)
                self.suffix_cache.stop_request(rid)
            return drafts

    return _KappaProposer


_cls = None


def KappaProposer(vllm_config):  # noqa: N802 - class-path loaded by vLLM
    """Factory kept callable as a class path by custom_class_proposer."""
    global _cls
    if _cls is None:
        _cls = _build_class()
    return _cls(vllm_config)
