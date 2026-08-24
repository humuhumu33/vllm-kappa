"""One engine boot against a κ-KV store — the isolatable unit of run_g3.

Run as a real file (multiprocessing spawn re-imports __main__; heredocs and
stdin scripts break EngineCore startup — measured T13).

Prints one JSON line: ttft, full_dt, stream, store index size before/after.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_g3 import DOC, run_boot  # noqa: E402


def index_count(store: str) -> int:
    p = Path(store, "index")
    return len(list(p.glob("*"))) if p.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--store", required=True)
    ap.add_argument("--repeats", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--connector", action="store_true")
    args = ap.parse_args()

    prompt = DOC * args.repeats + "\nQuestion: summarize the design in one sentence."
    before = index_count(args.store)
    ttft, full_dt, stream = run_boot(
        args.model, prompt, args.max_tokens, args.connector, args.store
    )
    print(
        json.dumps(
            {
                "ttft": ttft,
                "full_dt": full_dt,
                "stream": stream,
                "index_before": before,
                "index_after": index_count(args.store),
            }
        )
    )
    return 0


if __name__ == "__main__":
    main()
