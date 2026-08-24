"""End-to-end: seal a request into a store on disk, audit it with the CLI."""

import argparse

from verifier.kappa_verify import cmd_verify
from vllm_kappa.fabric import KappaStore, LocalStore
from vllm_kappa.witness import EngineFingerprint, build_chain

ENGINE = EngineFingerprint(
    vllm_sha="test",
    model_root="test-model",
    dtype="bfloat16",
    tp_size=1,
    block_size=16,
    batch_invariant=True,
)


def _populate(tmp_path, n=3):
    store = KappaStore(LocalStore(tmp_path))
    for k in range(n):
        records, seal = build_chain(
            ENGINE, {"temperature": 0.0}, list(range(k, k + 33)), list(range(20))
        )
        for r in records:
            store.put(r)
        store.put(seal)
    return store


def test_cli_verifies_store(tmp_path, capsys):
    _populate(tmp_path)
    rc = cmd_verify(argparse.Namespace(store=str(tmp_path), seal=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "3/3 seals verified" in out


def test_cli_refuses_tampered_store(tmp_path, capsys):
    store = _populate(tmp_path, n=1)
    # corrupt one witness record on disk (G1a, disk form)
    victim = next(
        p for p in store.local.root.glob("??/*") if p.stat().st_size < 200
    )
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    victim.write_bytes(bytes(data))
    rc = cmd_verify(argparse.Namespace(store=str(tmp_path), seal=None))
    assert rc == 1
