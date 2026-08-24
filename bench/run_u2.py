"""Gate U2/U2b — identity-hash overhead, measured on REAL block-key preimages.

Compares every mode vLLM ships (sha256-pickle default, sha256_cbor, xxhash,
xxhash_cbor) plus the U2b question: would blake3 over canonical CBOR beat
sha256? Preimages are exactly what `hash_block_tokens` hashes: 32-byte
parent digest, a full block of token ids, extra-keys tuple (None here).

Output: µs/block per mode per block size, and the per-request identity cost
at a representative 2k-token prompt — the number to weigh against end-to-end
serving cost.
"""

from __future__ import annotations

import json
import random
import statistics
import time

import blake3
import cbor2

from vllm.utils.hashing import sha256, sha256_cbor, xxhash, xxhash_cbor


def blake3_cbor(obj) -> bytes:
    return blake3.blake3(cbor2.dumps(obj, canonical=True)).digest()


MODES = {
    "sha256_pickle(default)": sha256,
    "sha256_cbor(=uor-addr)": sha256_cbor,
    "xxhash_pickle": xxhash,
    "xxhash_cbor": xxhash_cbor,
    "blake3_cbor(U2b)": blake3_cbor,
}


def bench(fn, preimages, reps=5):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for p in preimages:
            fn(p)
        times.append((time.perf_counter() - t0) / len(preimages) * 1e6)
    return round(statistics.median(times), 3)


def main():
    rng = random.Random(7)
    out = {"gate": "U2", "us_per_block": {}}
    for block_size in (16, 128, 256):
        parent = bytes(rng.randrange(256) for _ in range(32))
        preimages = [
            (parent, tuple(rng.randrange(150_000) for _ in range(block_size)), None)
            for _ in range(2000)
        ]
        out["us_per_block"][block_size] = {
            name: bench(fn, preimages) for name, fn in MODES.items()
        }
    # identity cost of a 2048-token prompt at vLLM CPU block size 128
    n_blocks = 2048 // 128
    out["per_2k_prompt_us"] = {
        name: round(out["us_per_block"][128][name] * n_blocks, 2)
        for name in MODES
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
