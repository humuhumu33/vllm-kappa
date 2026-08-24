"""U1 hard mode — byte-level conformance against the REAL uor-addr.

The ledger's re-derivation proof so far compared the mirror against vLLM
and against committed known-answer vectors. This closes the loop against
the reference implementation itself: `uor_addr.kappa.cbor_address` (the
Rust crate via its C ABI) must produce `sha256:<hex>` labels whose digest
equals `sha256(cbor2.dumps(obj, canonical=True))` for every identity shape
this program mints.

Known risk being tested (not assumed): uor-addr canonicalizes per RFC 8949
§4.2; cbor2's canonical mode follows RFC 7049 §3.9. The orderings coincide
for the shapes we mint (arrays; short-text-key maps) — this harness is the
proof, and any divergent shape is a finding, not a footnote.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cbor2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uor_addr import kappa  # noqa: E402

from vllm_kappa.addressing import NONE_HASH_SEED, none_hash  # noqa: E402

CASES = {
    # 1. the NONE_HASH root preimage
    "none_hash_seed": NONE_HASH_SEED,
    # 2. a first-block key preimage: (root, 16 token ids, no extras)
    "block_key_root": (none_hash(), tuple(range(16)), None),
    # 3. a chained block-key preimage with extra keys (mm id + offset)
    "block_key_extras": (
        b"\x11" * 32,
        tuple(range(100_000, 100_128)),
        (("mmhash-abc", 7),),
    ),
    # 4. a witness (`wit`) record as connector.py mints it
    "wit_record": {
        "v": 1,
        "t": "wit",
        "bk": b"\x22" * 32,
        "toks": list(range(16)),
        "prev": b"\x33" * 32,
    },
    # 5. a seal record shape (engine + sampling fingerprints: nested dicts)
    "seal_record": {
        "v": 1,
        "t": "seal",
        "eng": {
            "vllm_sha": "f94666b60",
            "model_root": "Qwen/Qwen2.5-0.5B-Instruct",
            "dtype": "torch.bfloat16",
            "tp_size": 1,
            "block_size": 128,
            "batch_invariant": False,
        },
        "samp": {"temperature": 0.0, "max_tokens": 48},
        "chain": [b"\x44" * 32, b"\x55" * 32],
    },
    # 6. a kvd record (KV byte digest binding)
    "kvd_record": {"v": 1, "t": "kvd", "bk": b"\x66" * 32, "kv": b"\x77" * 32},
}


def main() -> int:
    rows = {}
    ok = True
    for name, obj in CASES.items():
        enc = cbor2.dumps(obj, canonical=True)
        mine = "sha256:" + hashlib.sha256(enc).hexdigest()
        theirs = kappa.cbor_address(enc)
        rows[name] = {"match": mine == theirs, "label": theirs}
        ok &= mine == theirs
    print(json.dumps({"gate": "U1-conformance", "all_match": ok, "cases": rows}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
