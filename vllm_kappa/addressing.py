"""κ derivation. One identity, no new formats.

Block keys reproduce vLLM's prefix-cache hashing exactly (the byte-identity
with uor-addr's canonical CBOR address is the measured foundation of this
project — gate G0a). vLLM's own functions are preferred when importable; the
mirrors below exist so the verifier and tests run without a vLLM install,
and G0a asserts mirror ≡ vLLM whenever both are present.

Byte payloads (KV bytes, witness records) are digested with BLAKE3 and
labeled `blake3:<hex>` — the same label scheme hologram-fabric storage
serves objects under.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import blake3
import cbor2

# Mirrors vllm/v1/core/kv_cache_utils.py DEFAULT_NONE_HASH_SEED.
NONE_HASH_SEED = "vllm-none-hash"

BLOCK_CODEC = "sha256-cbor"
BYTES_CODEC = "blake3"


def sha256_cbor_mirror(obj: Any) -> bytes:
    """Exact mirror of vllm.utils.hashing.sha256_cbor."""
    return hashlib.sha256(cbor2.dumps(obj, canonical=True)).digest()


try:  # prefer vLLM's own implementation when present
    from vllm.utils.hashing import sha256_cbor as _vllm_sha256_cbor

    sha256_cbor = _vllm_sha256_cbor
except ImportError:  # verifier / test environments without vLLM
    sha256_cbor = sha256_cbor_mirror


def none_hash(seed: str = NONE_HASH_SEED) -> bytes:
    """Mirrors kv_cache_utils.init_none_hash for the sha256_cbor algorithm.

    vLLM lets PYTHONHASHSEED override the seed; witness verifiers must be
    given the same seed the serving instance resolved (it is recorded in the
    request seal's engine fingerprint).
    """
    return sha256_cbor(seed)


def block_key(
    parent: bytes | None,
    token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
    seed: str = NONE_HASH_SEED,
) -> bytes:
    """Mirrors kv_cache_utils.hash_block_tokens: hash((parent, tokens, extra))."""
    if not parent:
        parent = none_hash(seed)
    return sha256_cbor((parent, tuple(token_ids), extra_keys))


def chain_block_keys(
    token_ids: Sequence[int],
    block_size: int,
    extra_keys: tuple[Any, ...] | None = None,
    seed: str = NONE_HASH_SEED,
) -> list[bytes]:
    """Block keys for every FULL block of a token sequence, chained from root.

    Matches vLLM's convention: only full blocks are hashed; a trailing
    partial block has no key.
    """
    keys: list[bytes] = []
    parent: bytes | None = None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = block_key(
            parent, token_ids[start : start + block_size], extra_keys, seed
        )
        keys.append(parent)
    return keys


def blake3_digest(data: bytes) -> bytes:
    return blake3.blake3(data).digest()


def kappa(digest: bytes, codec: str) -> str:
    """The κ label: `<codec>:<hex>`. The only naming surface in this package."""
    return f"{codec}:{digest.hex()}"


def parse_kappa(label: str) -> tuple[str, bytes]:
    codec, _, hexdigest = label.partition(":")
    if not hexdigest:
        raise ValueError(f"not a κ label: {label!r}")
    return codec, bytes.fromhex(hexdigest)
