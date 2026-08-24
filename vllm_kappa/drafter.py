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

        # -- the hot path: delegate, then bookkeep -------------------------
        def propose(self, num_speculative_tokens, input_batch, sampled_token_ids, slot_mappings=None):
            for i, sampled_ids in enumerate(sampled_token_ids):
                if not sampled_ids:
                    continue
                req_id = input_batch.req_ids[i]
                entry = self._tracked.get(req_id)
                if entry is None:
                    index = input_batch.req_id_to_index[req_id]
                    num_prompt = input_batch.num_prompt_tokens[index]
                    start = max(0, num_prompt - self.max_tree_depth)
                    pattern = [
                        int(t)
                        for t in input_batch.token_ids_cpu[index, start:num_prompt]
                    ]
                    entry = (pattern, [])
                    self._tracked[req_id] = entry
                    self.counters["tracked_requests"] += 1
                entry[1].extend(int(t) for t in sampled_ids)

            drafts = super().propose(
                num_speculative_tokens, input_batch, sampled_token_ids, slot_mappings
            )

            for req_id in list(self._tracked.keys() - set(input_batch.req_ids)):
                self._seal(req_id)
            return drafts

    return _KappaProposer


_cls = None


def KappaProposer(vllm_config):  # noqa: N802 - class-path loaded by vLLM
    """Factory kept callable as a class path by custom_class_proposer."""
    global _cls
    if _cls is None:
        _cls = _build_class()
    return _cls(vllm_config)
