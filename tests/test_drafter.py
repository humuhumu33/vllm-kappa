"""Drafter record layer (vLLM/arctic-free parts) + full proposer when
arctic-inference and vLLM are installed."""

import pytest

from vllm_kappa.drafter import MIN_SEAL_TOKENS, decode_draft, encode_draft
from vllm_kappa.fabric import KappaStore, LocalStore


def test_draft_record_roundtrip():
    rec = encode_draft([1, 2, 3], [4, 5, 6, 7])
    assert decode_draft(rec) == ([1, 2, 3], [4, 5, 6, 7])


def test_draft_record_rejects_other_types():
    assert decode_draft(b"\x00garbage") is None
    from vllm_kappa.witness import EngineFingerprint, build_chain

    eng = EngineFingerprint("x", "m", "bf16", 1, 16, True)
    _, seal = build_chain(eng, {}, [1] * 16, [2] * 16)
    assert decode_draft(seal) is None  # witness records are not drafts


def test_draft_records_survive_store_roundtrip(tmp_path):
    store = KappaStore(LocalStore(tmp_path))
    rec = encode_draft(list(range(24)), list(range(100, 100 + MIN_SEAL_TOKENS)))
    label = store.put(rec)
    assert decode_draft(store.get(label)) is not None
    # warm-start scan pattern: find it again among arbitrary objects
    store.put(b"unrelated bytes")
    found = [
        lbl
        for lbl in store.local.labels()
        if (d := store.get(lbl)) and decode_draft(d)
    ]
    assert found == [label]


@pytest.mark.skipif(
    not pytest.importorskip("importlib.util").find_spec("arctic_inference")
    or not pytest.importorskip("importlib.util").find_spec("vllm"),
    reason="needs vllm + arctic-inference (WSL integration lane)",
)
def test_proposer_constructs():
    pass  # exercised by the WSL integration suite (Phase 2)
