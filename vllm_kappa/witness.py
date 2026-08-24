"""The κ-witness chain (L1).

Every record is one canonical-CBOR map; its identity is BLAKE3 over its own
bytes (the store label). A request seal binds: model root, engine fingerprint
(vLLM SHA, dtype, TP layout, batch-invariance flag, none-hash seed, block
size), canonical sampling params (T6: sampling is identity), the prompt, the
output tokens, and the ordered chain of block-witness record digests.

Verification recomputes everything from the token stream — block keys via the
same hash vLLM uses for its prefix cache — and refuses on the first byte that
does not re-derive. `witness.verify` in uor-addr replays a derivation trace
and never reads payloads (T1); this module is the payload-reading counterpart
and never claims otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cbor2

from .addressing import (
    NONE_HASH_SEED,
    blake3_digest,
    chain_block_keys,
    kappa,
    sha256_cbor,
    BYTES_CODEC,
)

SCHEMA_VERSION = 1


def _canonical(obj: Any) -> bytes:
    return cbor2.dumps(obj, canonical=True)


@dataclass(frozen=True)
class EngineFingerprint:
    vllm_sha: str
    model_root: str
    dtype: str
    tp_size: int
    block_size: int
    batch_invariant: bool
    none_hash_seed: str = NONE_HASH_SEED

    def to_map(self) -> dict:
        return {
            "vllm": self.vllm_sha,
            "model": self.model_root,
            "dtype": self.dtype,
            "tp": self.tp_size,
            "bs": self.block_size,
            "bi": self.batch_invariant,
            "seed": self.none_hash_seed,
        }

    @classmethod
    def from_map(cls, m: dict) -> "EngineFingerprint":
        return cls(
            vllm_sha=m["vllm"],
            model_root=m["model"],
            dtype=m["dtype"],
            tp_size=m["tp"],
            block_size=m["bs"],
            batch_invariant=m["bi"],
            none_hash_seed=m["seed"],
        )


def encode_block_witness(
    index: int,
    block_key_bytes: bytes,
    token_ids: list[int],
    kv_digest: bytes | None,
) -> bytes:
    return _canonical(
        {
            "v": SCHEMA_VERSION,
            "t": "bw",
            "i": index,
            "bk": block_key_bytes,
            "toks": token_ids,
            "kv": kv_digest,
        }
    )


def encode_seal(
    engine: EngineFingerprint,
    sampling: dict,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    chain_digests: list[bytes],
) -> bytes:
    return _canonical(
        {
            "v": SCHEMA_VERSION,
            "t": "seal",
            "eng": engine.to_map(),
            "samp": sampling,
            "prompt": prompt_token_ids,
            "out": output_token_ids,
            "chain": chain_digests,
        }
    )


def request_kappa(seal_bytes: bytes) -> str:
    """κ(request): the one label that names — and proves — this inference."""
    return kappa(blake3_digest(seal_bytes), BYTES_CODEC)


def build_chain(
    engine: EngineFingerprint,
    sampling: dict,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    kv_digests: list[bytes | None] | None = None,
) -> tuple[list[bytes], bytes]:
    """Returns ([block-witness record bytes...], seal record bytes)."""
    tokens = list(prompt_token_ids) + list(output_token_ids)
    keys = chain_block_keys(tokens, engine.block_size, seed=engine.none_hash_seed)
    records: list[bytes] = []
    for i, bk in enumerate(keys):
        toks = tokens[i * engine.block_size : (i + 1) * engine.block_size]
        kv = kv_digests[i] if kv_digests and i < len(kv_digests) else None
        records.append(encode_block_witness(i, bk, toks, kv))
    seal = encode_seal(
        engine,
        sampling,
        list(prompt_token_ids),
        list(output_token_ids),
        [blake3_digest(r) for r in records],
    )
    return records, seal


@dataclass
class VerifyReport:
    ok: bool
    request_kappa: str
    blocks_checked: int = 0
    kv_digests_present: int = 0
    problems: list[str] = field(default_factory=list)


def verify_chain(
    seal_bytes: bytes,
    fetch: Callable[[str], bytes | None],
) -> VerifyReport:
    """Recompute the whole chain from the token stream; refuse on any mismatch.

    `fetch` maps a κ label to record bytes (KappaStore.get verifies bytes
    against the label; any other fetch must do the same).
    """
    report = VerifyReport(ok=True, request_kappa=request_kappa(seal_bytes))

    def fail(msg: str) -> None:
        report.ok = False
        report.problems.append(msg)

    try:
        seal = cbor2.loads(seal_bytes)
    except Exception as e:
        fail(f"seal undecodable: {e}")
        return report
    if (
        not isinstance(seal, dict)
        or seal.get("v") != SCHEMA_VERSION
        or seal.get("t") != "seal"
    ):
        fail("not a v1 seal record")
        return report

    try:
        return _verify_decoded(seal, seal_bytes, fetch, report, fail)
    except Exception as e:  # a seal that cannot be walked is a refusal
        fail(f"malformed seal: {type(e).__name__}: {e}")
        return report


def _verify_decoded(seal, seal_bytes, fetch, report, fail) -> VerifyReport:
    engine = EngineFingerprint.from_map(seal["eng"])
    tokens = list(seal["prompt"]) + list(seal["out"])
    expected_keys = chain_block_keys(
        tokens, engine.block_size, seed=engine.none_hash_seed
    )
    if len(expected_keys) != len(seal["chain"]):
        fail(
            f"chain length {len(seal['chain'])} != "
            f"{len(expected_keys)} full blocks of the token stream"
        )

    for i, digest in enumerate(seal["chain"]):
        label = kappa(digest, BYTES_CODEC)
        rec_bytes = fetch(label)
        if rec_bytes is None:
            fail(f"block {i}: witness {label} unavailable")
            continue
        if blake3_digest(rec_bytes) != digest:
            fail(f"block {i}: witness bytes do not derive {label}")
            continue
        try:
            rec = cbor2.loads(rec_bytes)
        except Exception:
            rec = None
        if not isinstance(rec, dict):
            fail(f"block {i}: witness record undecodable")
            continue
        report.blocks_checked += 1
        if rec.get("i") != i:
            fail(f"block {i}: index mismatch ({rec.get('i')})")
        if i < len(expected_keys) and rec.get("bk") != expected_keys[i]:
            fail(f"block {i}: block key does not re-derive from token stream")
        start = i * engine.block_size
        if list(rec.get("toks", [])) != tokens[start : start + engine.block_size]:
            fail(f"block {i}: token slice mismatch")
        if rec.get("kv") is not None:
            report.kv_digests_present += 1

    # The sampling params are covered by the seal bytes themselves (the
    # request κ). Recompute a canonical re-encode to catch non-canonical
    # or reordered seals claiming the same identity.
    reencoded = encode_seal(
        engine,
        seal["samp"],
        list(seal["prompt"]),
        list(seal["out"]),
        list(seal["chain"]),
    )
    if reencoded != seal_bytes:
        fail("seal is not in canonical form")

    return report


def prompt_kappa(prompt_token_ids: list[int]) -> bytes:
    """Convenience: κ digest of a prompt alone (used by the drafter's seals)."""
    return sha256_cbor(tuple(prompt_token_ids))
