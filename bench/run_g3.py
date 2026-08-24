"""Gates G3a/G3b/G3c — κ-KV prefill reuse across engine boots.

Boot A (producer): serves a long-prefix request with VLLM_KAPPA_KV=1; every
full prompt block's KV is stored per block-κ.
Boot B (consumer): a FRESH process, same prompt — the connector reports the
κ-hit, injects stored blocks instead of prefilling, and generates.
Boot C (control): fresh process, no connector — honest cold prefill.

G3a: TTFT(B) vs TTFT(C) — the pulled-prefill win, plus the block hit count.
G3c: full token streams B ≡ C (same regime: single request, same shape).
G3b: one stored payload is corrupted on disk; Boot D must refuse the block
(verify-on-read), report it via load-errors so vLLM recomputes, and still
produce the correct stream.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

DOC = (
    "Design memo. The archive keeps every record under one content address "
    "derived from its bytes, so identical work is stored once and found by "
    "what it is rather than where it sits. Verification is recomputation of "
    "the address; a mismatch is a refusal. Costs fall as the table grows. "
)


def run_boot(model, prompt, max_tokens, use_connector, store):
    from vllm import LLM, SamplingParams

    kwargs = {}
    if use_connector:
        os.environ["VLLM_KAPPA_STORE_DIR"] = store
        os.environ["VLLM_KAPPA_KV"] = "1"
        kwargs["kv_transfer_config"] = {
            "kv_connector": "KappaConnector",
            "kv_connector_module_path": "vllm_kappa.connector",
            "kv_role": "kv_both",
        }
    else:
        os.environ.pop("VLLM_KAPPA_KV", None)
    memutil = float(os.environ.get("VLLM_KAPPA_MEMUTIL", "0.5"))
    llm = LLM(model=model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=memutil,
              enable_prefix_caching=False, **kwargs)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.perf_counter()
    llm.generate([prompt], sp1)
    ttft = time.perf_counter() - t0
    # NOTE: the TTFT request warms nothing for the full run below when
    # prefix caching is off; the second call re-prefills (or re-pulls).
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp)
    full_dt = time.perf_counter() - t0
    stream = list(outs[0].outputs[0].token_ids)
    del llm
    time.sleep(2)
    return round(ttft, 3), round(full_dt, 3), stream


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--repeats", type=int, default=24, help="doc paragraph repeats")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    store = os.path.expanduser("~/kappa-g3-store")
    shutil.rmtree(store, ignore_errors=True)
    prompt = DOC * args.repeats + "\nQuestion: summarize the design in one sentence."

    ttft_a, _, stream_a = run_boot(args.model, prompt, args.max_tokens, True, store)
    n_blocks = len(list(Path(store, "index").glob("*"))) if Path(store, "index").exists() else 0

    ttft_c, _, stream_c = run_boot(args.model, prompt, args.max_tokens, False, store)
    ttft_b, _, stream_b = run_boot(args.model, prompt, args.max_tokens, True, store)

    # G3b: corrupt one KV payload (the largest objects in the store)
    victim = max(Path(store).glob("??/*"), key=lambda p: p.stat().st_size)
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    victim.write_bytes(bytes(data))
    ttft_d, _, stream_d = run_boot(args.model, prompt, args.max_tokens, True, store)

    result = {
        "gate": "G3",
        "blocks_stored": n_blocks,
        "ttft_producer_cold": ttft_a,
        "ttft_control_cold": ttft_c,
        "ttft_kappa_pull": ttft_b,
        "ttft_tampered_fallback": ttft_d,
        "g3c_pull_identical": stream_b == stream_c,
        "g3b_tampered_identical": stream_d == stream_c,
        "producer_vs_control_identical": stream_a == stream_c,
    }
    print(json.dumps(result))
    return 0 if result["g3c_pull_identical"] and result["g3b_tampered_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
