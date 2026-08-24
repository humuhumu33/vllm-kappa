"""kappa-verify: replay a κ-witness chain and refuse anything that lies.

No vLLM, no GPU, no network needed — a seal + its witness records + this
file is a complete audit kit.

Usage:
  python -m verifier.kappa_verify verify --store DIR [--seal LABEL]
      Verify one seal (or every seal in the store). Exit 0 iff all pass.

  python -m verifier.kappa_verify selftest
      Build a synthetic chain, verify it, then flip one byte in every
      component in turn and assert each flip is refused (gate G1a's
      unit-level form).
"""

from __future__ import annotations

import argparse
import random
import sys

import cbor2

sys.path.insert(0, ".")  # allow running from the repo root without install

from vllm_kappa.fabric import IntegrityError, KappaStore, LocalStore  # noqa: E402
from vllm_kappa.witness import (  # noqa: E402
    EngineFingerprint,
    build_chain,
    request_kappa,
    verify_chain,
)


def _iter_seals(store: KappaStore):
    for label in store.local.labels():
        data = store.get(label)
        if data is None:
            continue
        try:
            rec = cbor2.loads(data)
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("t") == "seal":
            yield label, data


def cmd_verify(args: argparse.Namespace) -> int:
    store = KappaStore(LocalStore(args.store))
    seals = list(_iter_seals(store))
    if args.seal:
        seals = [(lbl, d) for lbl, d in seals if lbl == args.seal]
    if not seals:
        print("no seals found")
        return 1
    failures = 0
    for label, seal_bytes in seals:
        try:
            report = verify_chain(seal_bytes, store.get)
        except IntegrityError as e:
            print(f"REFUSED  {label}: {e}")
            failures += 1
            continue
        status = "OK      " if report.ok else "REFUSED "
        print(
            f"{status}{report.request_kappa}  "
            f"blocks={report.blocks_checked} kv={report.kv_digests_present}"
        )
        for p in report.problems:
            print(f"         - {p}")
        failures += 0 if report.ok else 1
    print(f"\n{len(seals) - failures}/{len(seals)} seals verified")
    return 1 if failures else 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    rng = random.Random(0)
    engine = EngineFingerprint(
        vllm_sha="selftest",
        model_root="selftest-model",
        dtype="bfloat16",
        tp_size=1,
        block_size=16,
        batch_invariant=True,
    )
    prompt = [rng.randrange(32000) for _ in range(40)]
    output = [rng.randrange(32000) for _ in range(25)]
    records, seal = build_chain(engine, {"temperature": 0.0}, prompt, output)
    by_label = {request_kappa(r): r for r in records}

    from vllm_kappa.addressing import BYTES_CODEC, blake3_digest, kappa

    def fetch(label: str):
        return by_label.get(label)

    by_label = {kappa(blake3_digest(r), BYTES_CODEC): r for r in records}

    report = verify_chain(seal, fetch)
    assert report.ok, f"clean chain failed: {report.problems}"
    print(f"clean chain OK: {report.request_kappa} ({report.blocks_checked} blocks)")

    refused = 0
    trials = 0
    # tamper every byte position of every witness record and of the seal
    for name, data in [("seal", seal)] + [
        (f"witness[{i}]", r) for i, r in enumerate(records)
    ]:
        for pos in range(len(data)):
            trials += 1
            flipped = bytearray(data)
            flipped[pos] ^= 0xFF
            flipped = bytes(flipped)
            if name == "seal":
                # A flipped seal is either internally inconsistent, or a
                # DIFFERENT valid seal — in which case its κ no longer
                # matches the claimed one. Both are refusals: the auditor
                # always holds the claimed κ (the store fetch enforces it).
                claimed = request_kappa(seal)
                r = verify_chain(flipped, fetch)
                ok = (not r.ok) or (r.request_kappa != claimed)
            else:
                i = int(name[8:-1])
                mutated = dict(by_label)
                lbl = kappa(blake3_digest(records[i]), BYTES_CODEC)
                mutated[lbl] = flipped  # served bytes no longer match label
                r = verify_chain(seal, mutated.get)
                ok = not r.ok
            refused += ok
    print(f"tamper trials: {refused}/{trials} refused")
    if refused != trials:
        print("SELFTEST FAILED: some tampering was accepted")
        return 1
    print("selftest OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="kappa-verify")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--store", required=True)
    v.add_argument("--seal")
    v.set_defaults(fn=cmd_verify)
    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
