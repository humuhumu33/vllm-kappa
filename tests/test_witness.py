"""L1 unit gates: chain build/verify, canonical-form enforcement, and the
unit form of G1a — every single-byte tamper is refused."""

import random

import cbor2
import pytest

from vllm_kappa.addressing import BYTES_CODEC, blake3_digest, kappa
from vllm_kappa.witness import (
    EngineFingerprint,
    build_chain,
    encode_seal,
    request_kappa,
    verify_chain,
)

ENGINE = EngineFingerprint(
    vllm_sha="test",
    model_root="test-model",
    dtype="bfloat16",
    tp_size=1,
    block_size=16,
    batch_invariant=True,
)


def _chain(n_prompt=40, n_out=25, seed=7):
    rng = random.Random(seed)
    prompt = [rng.randrange(32000) for _ in range(n_prompt)]
    out = [rng.randrange(32000) for _ in range(n_out)]
    records, seal = build_chain(ENGINE, {"temperature": 0.0}, prompt, out)
    fetchmap = {kappa(blake3_digest(r), BYTES_CODEC): r for r in records}
    return records, seal, fetchmap


def test_clean_chain_verifies():
    records, seal, fetchmap = _chain()
    report = verify_chain(seal, fetchmap.get)
    assert report.ok, report.problems
    assert report.blocks_checked == len(records) == (40 + 25) // 16
    assert report.request_kappa == request_kappa(seal)


def test_output_token_tamper_refused():
    _, seal, fetchmap = _chain()
    s = cbor2.loads(seal)
    s["out"][3] ^= 1
    forged = cbor2.dumps(s, canonical=True)
    report = verify_chain(forged, fetchmap.get)
    assert not report.ok  # block keys no longer re-derive


def test_witness_byte_tamper_refused():
    records, seal, fetchmap = _chain()
    label = next(iter(fetchmap))
    flipped = bytearray(fetchmap[label])
    flipped[len(flipped) // 2] ^= 0xFF
    fetchmap[label] = bytes(flipped)
    report = verify_chain(seal, fetchmap.get)
    assert not report.ok


def test_missing_witness_refused():
    _, seal, fetchmap = _chain()
    fetchmap.pop(next(iter(fetchmap)))
    report = verify_chain(seal, fetchmap.get)
    assert not report.ok


def test_sampling_is_identity():
    r1, s1 = build_chain(ENGINE, {"temperature": 0.0}, [1] * 16, [2] * 16)
    r2, s2 = build_chain(ENGINE, {"temperature": 0.7}, [1] * 16, [2] * 16)
    assert request_kappa(s1) != request_kappa(s2)  # T6


def test_engine_is_identity():
    other = EngineFingerprint(
        vllm_sha="test",
        model_root="DIFFERENT-model",
        dtype="bfloat16",
        tp_size=1,
        block_size=16,
        batch_invariant=True,
    )
    _, s1 = build_chain(ENGINE, {}, [1] * 16, [2] * 16)
    _, s2 = build_chain(other, {}, [1] * 16, [2] * 16)
    assert request_kappa(s1) != request_kappa(s2)


def test_non_canonical_seal_refused():
    records, seal, fetchmap = _chain()
    decoded = cbor2.loads(seal)
    non_canonical = cbor2.dumps(decoded, canonical=False)
    if non_canonical == seal:
        pytest.skip("encoder produced canonical form anyway")
    report = verify_chain(non_canonical, fetchmap.get)
    assert not report.ok


def test_chain_length_mismatch_refused():
    _, seal, fetchmap = _chain()
    s = cbor2.loads(seal)
    s["chain"] = s["chain"][:-1]
    forged = encode_seal(
        # rebuild canonically so only the length lie remains
        __import__("vllm_kappa.witness", fromlist=["EngineFingerprint"])
        .EngineFingerprint.from_map(s["eng"]),
        s["samp"],
        s["prompt"],
        s["out"],
        s["chain"],
    )
    report = verify_chain(forged, fetchmap.get)
    assert not report.ok
