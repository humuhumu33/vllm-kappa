"""Ordered block keys for the fixed G3/G1b workload prompt.

The KeyedIndex names files by key hex, which does not encode block order;
re-derive the ordered chain from the same prompt g3_boot uses, with the
engine block size read from any stored kvd/seal... kept simple: tokenize
the fixed prompt with the model's tokenizer and chain with block size 128
(the CPU backend's cache_config.block_size — measured 2026-08-24).
"""

from __future__ import annotations

from functools import lru_cache

from run_g3 import DOC
from vllm_kappa.addressing import chain_block_keys

BLOCK_SIZE = 128
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=1)
def ordered_keys(store_dir: str, repeats: int = 24) -> list[str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = DOC * repeats + "\nQuestion: summarize the design in one sentence."
    ids = tok(prompt).input_ids
    usable = ((len(ids) - 1) // BLOCK_SIZE) * BLOCK_SIZE
    return [k.hex() for k in chain_block_keys(ids[:usable], BLOCK_SIZE)]
