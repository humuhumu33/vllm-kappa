"""Gate G0b — batch-invariance on this hardware.

Runs the same prompts through vLLM under different batch compositions and
process restarts, and demands bit-identical greedy token streams and
per-token logprob float bits. Bit-level logit/KV comparison is the full
form; this harness is the necessary condition that catches any invariance
break visible through the sampling path.

Run (inside the serving lane venv):
    VLLM_BATCH_INVARIANT=1 python bench/run_g0b.py --model Qwen/Qwen2.5-0.5B-Instruct
Then rerun the process (the harness prints a digest; two runs must match).

Exit 0 iff every batch composition yields identical streams AND the run
digest matches --expect (when given).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--expect", help="digest from a previous run (restart check)")
    args = ap.parse_args()

    if os.getenv("VLLM_BATCH_INVARIANT") != "1":
        print("refusing: VLLM_BATCH_INVARIANT must be 1 for G0b")
        return 2

    from vllm import LLM, SamplingParams

    prompts = [
        f"Q{i}: List three facts about the number {i * 37 % 100}."
        for i in range(args.n_prompts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True)

    def run(batch: list[str]) -> dict[str, bytes]:
        outs = llm.generate(batch, sp)
        result = {}
        for prompt, out in zip(batch, outs):
            seq = out.outputs[0]
            blob = bytes()
            for tok, lps in zip(seq.token_ids, seq.logprobs or []):
                lp = lps[tok].logprob
                blob += struct.pack("<i", tok) + struct.pack("<f", lp)
            result[prompt] = blob
        return result

    # composition A: all prompts in one batch; B: singles; C: shuffled pairs
    ref = run(prompts)
    compositions = {
        "singles": [[p] for p in prompts],
        "pairs-shuffled": [
            prompts[i::2] for i in range(2)
        ],
        "reversed": [list(reversed(prompts))],
    }

    ok = True
    for name, batches in compositions.items():
        got: dict[str, bytes] = {}
        for b in batches:
            got.update(run(b))
        mismatched = [p for p in prompts if got[p] != ref[p]]
        status = "OK" if not mismatched else f"FAIL ({len(mismatched)} prompts)"
        print(f"composition {name:>14}: {status}")
        ok &= not mismatched

    digest = hashlib.sha256(
        b"".join(ref[p] for p in prompts)
    ).hexdigest()
    print(json.dumps({"gate": "G0b", "within_process": ok, "digest": digest}))

    if args.expect and args.expect != digest:
        print(f"restart check FAIL: digest != --expect {args.expect}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
