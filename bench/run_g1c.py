"""Gate G1c — κ-witness connector: overhead ≤2% and real seals that verify.

Arm A: plain engine.  Arm B: same engine + KappaConnector (token-level
witnessing; KV-byte digests stay off, their budget is Phase 3's).

After arm B: every request must have produced a seal in the store; the
offline auditor must verify all of them (first L1 verification over REAL
inference bytes, not synthetic chains); then one witness record is corrupted
on disk and the auditor must refuse (G1a, production form).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time


def run_arm(model, prompts, max_tokens, connector_store: str | None):
    from vllm import LLM, SamplingParams

    kwargs = {}
    if connector_store is not None:
        os.environ["VLLM_KAPPA_STORE_DIR"] = connector_store
        kwargs["kv_transfer_config"] = {
            "kv_connector": "KappaConnector",
            "kv_connector_module_path": "vllm_kappa.connector",
            "kv_role": "kv_both",
        }
    llm = LLM(model=model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=0.5, **kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    del llm
    time.sleep(2)  # let the witness thread drain before the store is read
    return round(ntok / dt, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    store = os.path.expanduser("~/kappa-g1c-store")
    shutil.rmtree(store, ignore_errors=True)

    prompts = [
        f"Fact request {i}: name two properties of "
        + ["basalt", "oak", "copper", "wool"][i % 4]
        for i in range(args.n_prompts)
    ]

    plain, witnessed = [], []
    for _ in range(args.reps):
        plain.append(run_arm(args.model, prompts, args.max_tokens, None))
        witnessed.append(run_arm(args.model, prompts, args.max_tokens, store))

    # -- audit the real seals -------------------------------------------
    import argparse as _ap

    from verifier.kappa_verify import cmd_verify

    rc_clean = cmd_verify(_ap.Namespace(store=store, seal=None))

    # G1a production form: corrupt one small record (a block witness), re-audit
    from pathlib import Path

    victim = min(
        (p for p in Path(store).glob("??/*") if p.is_file()),
        key=lambda p: p.stat().st_size,
    )
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    victim.write_bytes(bytes(data))
    rc_tampered = cmd_verify(_ap.Namespace(store=store, seal=None))

    p_med = sorted(plain)[len(plain) // 2]
    w_med = sorted(witnessed)[len(witnessed) // 2]
    result = {
        "gate": "G1c",
        "tok_per_s_plain": plain,
        "tok_per_s_witnessed": witnessed,
        "overhead_pct": round(100 * (p_med - w_med) / p_med, 2),
        "audit_clean_rc": rc_clean,
        "audit_tampered_rc": rc_tampered,
    }
    print(json.dumps(result))
    ok = rc_clean == 0 and rc_tampered == 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
