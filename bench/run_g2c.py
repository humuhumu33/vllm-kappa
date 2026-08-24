"""Gate G2c — controlled drafter throughput, three arms, two boots each.

Arms:  none   — no speculative decoding (baseline)
       suffix — vLLM's in-tree Arctic suffix decoding (method:"suffix");
                per-process memory only
       kappa  — the κ-drafter (custom_class); persistent shared store

Each arm runs the same workload in TWO consecutive engine boots. The κ-arm's
second boot warm-starts from the store its first boot sealed — that
cross-boot memory is precisely the delta over in-tree suffix decoding, so
the headline comparison is boot2(kappa) vs boot2(suffix) on the repeat
workload.

Workloads: repeat — an agentic-replay shape: a small set of templated
           requests, each issued several times (RAG boilerplate, re-run
           agents); novel — unique prompts, no self-similarity (G2d guard:
           drafter must cost ≤3%).

Control gate: a fixed numpy GEMM is timed before every boot; reps whose
control drifts >25% from the session median are flagged (machine busy) and
the JSON says so — Hoefler rule: never publish an unflagged noisy number.

Byte-identity is asserted arm-vs-baseline on every boot (G2a continuous).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time


def control_gflops() -> float:
    import numpy as np

    n = 1024
    a = np.random.default_rng(0).standard_normal((n, n), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(8):
        a = a @ a
        a /= np.abs(a).max() + 1.0
    dt = time.perf_counter() - t0
    return round(8 * 2 * n**3 / dt / 1e9, 2)


def make_workload(kind: str, n: int) -> list[str]:
    topics = ["rivers", "compilers", "chess", "bread", "tides", "glaciers"]
    if kind == "repeat":
        base = [
            "You are a support agent. Policy: refunds within 30 days, store "
            f"credit after. Customer asks about {t}. Answer in two sentences."
            for t in topics
        ]
        return [base[i % len(base)] for i in range(n)]
    return [
        f"Question {i * 7919 % 1000}: explain {topics[i % 6]} to a "
        f"{['pilot', 'chef', 'nurse', 'farmer'][i % 4]} in two sentences."
        for i in range(n)
    ]


def run_boot(model, prompts, max_tokens, arm, num_spec):
    from vllm import LLM, SamplingParams

    spec = None
    if arm == "suffix":
        spec = {"method": "suffix", "num_speculative_tokens": num_spec}
    elif arm == "kappa":
        spec = {
            "method": "custom_class",
            "model": "vllm_kappa.drafter.KappaProposer",
            "num_speculative_tokens": num_spec,
        }
    kwargs = {"speculative_config": spec} if spec else {}
    llm = LLM(model=model, dtype="bfloat16", enforce_eager=True, **kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    streams = [tuple(o.outputs[0].token_ids) for o in outs]
    ntok = sum(len(s) for s in streams)
    del llm
    return streams, round(ntok / dt, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--num-spec", type=int, default=8)
    ap.add_argument("--workloads", default="repeat,novel")
    args = ap.parse_args()

    store_root = os.environ.get(
        "VLLM_KAPPA_STORE_ROOT", os.path.expanduser("~/kappa-g2c")
    )

    results = []
    controls = []
    for kind in args.workloads.split(","):
        prompts = make_workload(kind, args.n_prompts)
        reference = None
        for arm in ("none", "suffix", "kappa"):
            store = os.path.join(store_root, f"{kind}-{arm}")
            shutil.rmtree(store, ignore_errors=True)
            os.environ["VLLM_KAPPA_STORE_DIR"] = store
            for boot in (1, 2):
                ctrl = control_gflops()
                controls.append(ctrl)
                streams, tps = run_boot(
                    args.model, prompts, args.max_tokens, arm, args.num_spec
                )
                if arm == "none" and boot == 1:
                    reference = streams
                ident = (
                    sum(a == b for a, b in zip(streams, reference))
                    if reference
                    else None
                )
                row = {
                    "workload": kind,
                    "arm": arm,
                    "boot": boot,
                    "tok_per_s": tps,
                    "identical_to_ref": ident,
                    "control_gflops": ctrl,
                }
                results.append(row)
                print(json.dumps(row), flush=True)

    med = sorted(controls)[len(controls) // 2]
    for r in results:
        r["control_flag"] = abs(r["control_gflops"] - med) / med > 0.25
    print(json.dumps({"gate": "G2c", "control_median_gflops": med,
                      "rows": results}))
    bad = [
        r
        for r in results
        if r["identical_to_ref"] is not None
        and r["identical_to_ref"] != args.n_prompts
    ]
    for r in bad:
        print(f"IDENTITY BREAK: {r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
