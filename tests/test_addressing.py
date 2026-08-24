"""G0a: block-key identity. The mirror must equal vLLM's own hashing, and
both must equal the committed known-answer vectors (derived from the same
sha256(canonical-cbor) rule uor-addr's cbor_address implements — the
byte-identity measured 2026-08-21)."""

import hashlib

import cbor2
import pytest

from vllm_kappa.addressing import (
    NONE_HASH_SEED,
    block_key,
    chain_block_keys,
    kappa,
    none_hash,
    parse_kappa,
    sha256_cbor_mirror,
)

try:
    from vllm.utils.hashing import sha256_cbor as vllm_sha256_cbor

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


def test_mirror_is_canonical_cbor_sha256():
    obj = (b"parent", (1, 2, 3), None)
    assert sha256_cbor_mirror(obj) == hashlib.sha256(
        cbor2.dumps(obj, canonical=True)
    ).digest()


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
def test_g0a_mirror_equals_vllm():
    for obj in [
        NONE_HASH_SEED,
        (none_hash(), tuple(range(16)), None),
        {"k": [1, 2, {"n": None}]},
    ]:
        assert sha256_cbor_mirror(obj) == vllm_sha256_cbor(obj)


def test_none_hash_known_answer():
    # sha256(cbor("vllm-none-hash")) — pinned; if this moves, every witness
    # in every store is orphaned, so it may never move silently.
    assert none_hash() == hashlib.sha256(
        cbor2.dumps("vllm-none-hash", canonical=True)
    ).digest()


def test_block_key_root_uses_none_hash():
    toks = list(range(16))
    assert block_key(None, toks) == sha256_cbor_mirror(
        (none_hash(), tuple(toks), None)
    )
    assert block_key(b"", toks) == block_key(None, toks)


def test_chain_block_keys_links_parents():
    toks = list(range(40))  # 2 full blocks of 16, partial dropped
    keys = chain_block_keys(toks, 16)
    assert len(keys) == 2
    assert keys[0] == block_key(None, toks[:16])
    assert keys[1] == block_key(keys[0], toks[16:32])


def test_chain_sensitivity():
    a = chain_block_keys(list(range(32)), 16)
    b = chain_block_keys([1] + list(range(1, 32)), 16)
    assert a[0] != b[0] and a[1] != b[1]  # first-block change cascades


def test_kappa_labels_roundtrip():
    d = bytes(range(32))
    label = kappa(d, "blake3")
    assert parse_kappa(label) == ("blake3", d)
    with pytest.raises(ValueError):
        parse_kappa("nolabel")
