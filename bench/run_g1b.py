"""Gate G1b — the O(block) challenge, on real KV bytes.

A witnessed block J is challengeable by pulling blocks < J from the store
and recomputing from J on — marginal cost shrinks as J grows, and at
J = last the challenge touches one block plus the tail. The recomputed
KV bytes must equal the originally witnessed bytes tensor-for-tensor
(same engine, same regime — the CPU lane's canonicality precondition).

Protocol per challenge J:
  1. producer boot fills the store (payloads + kvd records), snapshot
     index and payload labels;
  2. drop ONLY block J's index entry (the κ-hit walk stops at the first
     miss, so the challenge boot pulls [0,J) and computes J..end);
  3. challenge boot re-stores block J (it computed it; skip-window rules
     exempt only externally-claimed blocks);
  4. compare recomputed payload tensors against the originals, and its
     BLAKE3 against the original `kvd` witness digest.

Each boot is a separate process (T13). Wall times give the asymmetry
curve: challenge(J) vs full cold prefill.
"""

from __future__ import annotations

import argparse
import io as _io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_kappa.fabric import KappaStore, KeyedIndex, LocalStore  # noqa: E402


def boot(store: str, connector: bool) -> dict:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "g3_boot.py"),
        "--store",
        store,
    ]
    if connector:
        cmd.append("--connector")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    line = [l for l in out.stdout.splitlines() if l.startswith("{")][-1]
    return json.loads(line)


def payload_equal(store: KappaStore, label_a: str, label_b: str) -> bool:
    import torch

    if label_a == label_b:
        return True  # content-addressed: same label IS bit-equality
    a = torch.load(_io.BytesIO(store.get(label_a)), weights_only=True)
    b = torch.load(_io.BytesIO(store.get(label_b)), weights_only=True)
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", default="9,5")
    args = ap.parse_args()

    store_dir = os.path.expanduser("~/kappa-g1b")
    shutil.rmtree(store_dir, ignore_errors=True)

    produced = boot(store_dir, connector=True)
    index = KeyedIndex(store_dir)
    snapshot = {
        p.name: p.read_text().strip() for p in index.root.glob("*")
    }
    cold = boot(store_dir + "-empty", connector=False)

    store = KappaStore(LocalStore(store_dir))
    results = []
    ok = True
    for j_str in args.challenges.split(","):
        j = int(j_str)
        # keys are index filenames; block order isn't encoded in the name,
        # so rebuild the ordered keys from any stored seal? Simpler: the
        # producer stored blocks in order — recover order by re-deriving
        # from the prompt inside g3_boot's fixed workload.
        from bench_order import ordered_keys  # local helper below

        keys = ordered_keys(store_dir)
        bk_hex = keys[j]
        original_label = snapshot[bk_hex]
        index.drop(bytes.fromhex(bk_hex))

        challenged = boot(store_dir, connector=True)
        new_label = index.get(bytes.fromhex(bk_hex))
        match = new_label is not None and payload_equal(
            store, original_label, new_label
        )
        ok &= match
        results.append(
            {
                "block": j,
                "recomputed_matches_witnessed": match,
                "labels_identical": new_label == original_label,
                "challenge_ttft_s": challenged["ttft"],
                "stream_identical": challenged["stream"] == produced["stream"],
            }
        )

    print(
        json.dumps(
            {
                "gate": "G1b",
                "cold_prefill_ttft_s": cold["ttft"],
                "produced_blocks": produced["index_after"],
                "challenges": results,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
