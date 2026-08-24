"""Gate G2a — verified-by-construction: κ-drafter on vs off, greedy, byte-identical.

This gate IS the security argument for L2: speculative decoding's target-model
verification guarantees the drafter cannot change one output token, only
latency. It also exercises G2b when the store is pre-poisoned (--poison).

Run twice for the persistence demo (G2c's restart half): the second process
warm-starts its suffix tree from the κ-store written by the first.

CPU-lane caveat (measured in G0b-native, 2026-08-24): the CPU backend is not
bit-invariant at batch size 1, so rare greedy tie-flips between the spec-decode
verify pass (m=k) and plain decode (m=1) are possible for reasons unrelated to
the drafter. Mismatches are therefore reported per-token with logprob context;
tie-flips (near-zero logprob gap) are counted separately from real divergences.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def run_generate(model: str, prompts: list[str], max_tokens: int, spec: dict | None):
    from vllm import LLM, SamplingParams

    kwargs = {}
    if spec is not None:
        kwargs["speculative_config"] = spec
    llm = LLM(
        model=model,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.5,  # CPU backend: fraction of RAM reserved
        disable_log_stats=False,
        **kwargs,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=2)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    streams = {}
    for p, o in zip(prompts, outs):
        seq = o.outputs[0]
        toks = list(seq.token_ids)
        lps = []
        for tok, lp in zip(toks, seq.logprobs or []):
            entry = sorted(lp.values(), key=lambda e: -e.logprob)
            gap = entry[0].logprob - entry[1].logprob if len(entry) > 1 else 99.0
            lps.append((lp[tok].logprob, gap))
        streams[p] = (toks, lps)
    ntok = sum(len(v[0]) for v in streams.values())
    del llm
    return streams, dt, ntok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--num-spec", type=int, default=8)
    ap.add_argument("--poison", action="store_true", help="G2b: pre-poison the store")
    args = ap.parse_args()

    store_dir = os.environ.setdefault("VLLM_KAPPA_STORE_DIR", "/tmp/kappa-g2a")

    if args.poison:
        import random

        from vllm_kappa.drafter import encode_draft
        from vllm_kappa.fabric import KappaStore, LocalStore

        rng = random.Random(666)
        store = KappaStore(LocalStore(store_dir))
        for _ in range(200):
            store.put(
                encode_draft(
                    [rng.randrange(150000) for _ in range(16)],
                    [rng.randrange(150000) for _ in range(24)],
                )
            )
        print(f"poisoned store with 200 garbage draft records at {store_dir}")

    prompts = [
        f"Write exactly two sentences about topic {i}: "
        + ["rivers", "compilers", "chess", "bread", "tides",
           "glaciers", "violins", "beetles"][i % 8]
        for i in range(args.n_prompts)
    ]

    base, t_base, n_base = run_generate(args.model, prompts, args.max_tokens, None)
    spec_cfg = {
        "method": "custom_class",
        "model": "vllm_kappa.drafter.KappaProposer",
        "num_speculative_tokens": args.num_spec,
    }
    draft, t_draft, n_draft = run_generate(
        args.model, prompts, args.max_tokens, spec_cfg
    )

    identical = 0
    tie_flips = 0
    real_divergences = 0
    for p in prompts:
        b_toks, b_lps = base[p]
        d_toks, _ = draft[p]
        if b_toks == d_toks:
            identical += 1
            continue
        # first divergence point: tie-flip or real?
        i = next(k for k, (a, b) in enumerate(zip(b_toks, d_toks)) if a != b)
        gap = b_lps[i][1] if i < len(b_lps) else 99.0
        if gap < 1e-3:
            tie_flips += 1
        else:
            real_divergences += 1
            print(f"REAL DIVERGENCE at token {i} (logprob gap {gap:.5f}): {p[:40]}")

    from vllm_kappa.drafter import decode_draft
    from vllm_kappa.fabric import KappaStore, LocalStore

    check = KappaStore(LocalStore(store_dir))
    n_records = sum(
        1
        for lbl in check.local.labels()
        if (d := check.get(lbl)) and decode_draft(d)
    )

    result = {
        "gate": "G2a",
        "draft_records_in_store": n_records,
        "identical": identical,
        "tie_flips": tie_flips,
        "real_divergences": real_divergences,
        "n_prompts": args.n_prompts,
        "poisoned": args.poison,
        "tok_per_s_base": round(n_base / t_base, 2),
        "tok_per_s_draft": round(n_draft / t_draft, 2),
        "store_dir": store_dir,
    }
    print(json.dumps(result))
    return 0 if real_divergences == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
