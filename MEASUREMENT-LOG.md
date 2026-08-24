# Measurement log

Every run recorded, rejected runs included (PROMPT.md §4.6). Machine for local
entries: Ryzen AI MAX 390 (Zen 5), Windows 11, Python 3.11 — the **unit lane**.
Serving gates (G0b, G1c, G2*, G3*) require the WSL/Linux integration lane and
are marked PENDING until run there.

## 2026-08-24 — Phase 0/1 unit lane

| Gate | Status | Evidence |
|---|---|---|
| G0a (identity) | **PASS (unit form)** | `tests/test_addressing.py`: mirror ≡ sha256(canonical-CBOR) known-answer vectors incl. NONE_HASH = H("vllm-none-hash"); mirror ≡ `vllm.utils.hashing.sha256_cbor` asserted whenever vLLM importable (skipped on this host — vLLM does not install on native Windows). Byte-identity of this rule with uor-addr `cbor_address` was measured 2026-08-21 (vllm-uor-research, gate0). |
| G1a (tamper) | **PASS (unit form)** | `verifier selftest`: 900/900 single-byte flips refused across every witness record and the seal (240924 run: initial 896/900 exposed that flips in free seal fields produce *different valid seals* — refusal is by claimed-κ mismatch, the form the store fetch enforces; selftest now models the claim). Disk form: `tests/test_verifier_cli.py` corrupts a stored record, CLI exits 1. |
| G0b (determinism) | PENDING | needs serving lane |
| G1b (O(block) asymmetry) | PENDING | needs Phase 3 load path |
| G1c (witness overhead) | PENDING | needs serving lane |

Findings while testing (kept per honesty discipline):
- `cbor2.loads` on tampered bytes can return non-dict scalars without raising;
  both decoders originally assumed dicts and crashed instead of refusing.
  Fixed with type guards; crash-paths now covered by the 900-flip sweep.
- Refusal semantics clarified: internal consistency alone is not the boundary —
  the **claimed κ** is. A verifier must always compare `report.request_kappa`
  against the label the seal was fetched under (KappaStore does this
  automatically; auditors using other channels must not skip it).

Unit suite: 24 passed, 2 skipped (vllm / arctic-inference not present in this
lane) — `python -m pytest tests/ -q`.

## 2026-08-24 — serving-lane bring-up (WSL Ubuntu, CPU backend)

Lane re-pin: no CPU wheel exists for `1baf372b` (404 at the per-commit index);
lane runs the nightly CPU wheel at **f94666b60d4c58ec0807d22c837cfae322a1dde9**
(vllm 0.26.1rc1.dev1133+gf94666b60.cpu). `assert_seams()` guards the move at
import; the Windows reference clone stays at the original pin for source reads.

Environment incidents (all one root cause — **host C: drive at 0 bytes free**,
the WSL VHD and pagefile live there):
- uv "Bus error", pip "OSError: Errno 5", WSL `E_UNEXPECTED` catastrophic
  failures and `0x80072746` connection resets were all disk-full symptoms.
- Writes during the outage silently truncated installed files
  (`typing_extensions.py` and others at 0 bytes) → venv rebuilt from scratch;
  never trust a venv that lived through ENOSPC.
- Recovered ~20 GB: pip/npm caches, >7-day temp, `.holo-ship-tmp`,
  `_releases` pruned to newest (holo status verified healthy after),
  8.7 GB pip/uv caches inside the VHD, `fstrim` via `wsl -u root` (874 GiB
  trimmed; VHD compaction still pending an elevated `diskpart compact vdisk`).
- `.wslconfig` swap reduced 16 GB → 6 GB (backup at `.wslconfig.bak-kappa`):
  a 16 GB swap vhdx cannot coexist with <16 GB free on its host drive.
- arctic-inference 0.2.0 source build initially failed: venv `ninja` shim
  truncated (ENOSPC casualty), then missing `python3.12-dev` headers —
  installed via `wsl -u root` (no sudo password needed there). Final recipe:
  `pip install nanobind cmake ninja` then `--no-build-isolation` (isolation
  re-downloads a multi-GB torch and its write burst was killing WSL).
- A whole "successful" vllm install turned out to be 2,279 zero-byte files —
  ext4 delayed allocation dropped every unsynced data block across the hard
  kills. Retraction: an earlier "seam moved" conclusion was this artifact
  (importing an empty hashing.py raises the same ImportError). Rule adopted:
  `sync` after every install step; distrust any venv that lived through a
  WSL hard kill.

## 2026-08-24 — G0a (full) and G0b, CPU lane

Lane: WSL Ubuntu, Zen 5 (12 cores), vllm 0.26.1rc1.dev1133+gf94666b60.cpu,
Qwen2.5-0.5B-Instruct bf16, eager, block_size 16 default.

| Gate | Status | Evidence |
|---|---|---|
| G0a (identity, full form) | **PASS** | `sha256_cbor` ≡ mirror byte-for-byte against the real vllm at f94666b60; all four seams present (`assert_seams()` clean); arctic `SuffixDecodingCache` importable. |
| G0b (flag mode) | **BLOCKED on CPU — structural** | `VLLM_BATCH_INVARIANT=1` monkey-patches mean/matmul with **Triton kernels** (`mean_kernel[grid]` → "'function' object is not subscriptable" with 0 GPU drivers). Batch-invariant mode is GPU-only; full canonical bytes require the GPU lane. |
| G0b (native invariance) | **PARTIAL — precisely characterized** | 12 prompts × 32 tok, token ids + logprob float bits: reference batch-of-12 vs **reversed** order: bit-identical ✔; vs two **batches of 6**: bit-identical ✔; vs **singles** (batch-of-1): 12/12 diverge ✗ (oneDNN m=1 matvec path ≠ m≥2 GEMM path). **Restart**: digest `b4dc9bcf…` byte-identical across two engine boots ✔. |

Consequence (PROMPT.md §7 re-scope): on the CPU lane, κ(request) → bytes is
re-derivable **per batching regime** — witness verification must replay
solo-as-solo / batched-as-batched. Order and moderate batch-size changes do
not move a bit; only the m=1 kernel boundary does. Regime-free canonical
bytes await a GPU lane with `VLLM_BATCH_INVARIANT`.

## 2026-08-24 — G2a / G2b / persistence, first light (CPU lane)

KappaProposer live inside a real engine via `method:"custom_class"` (nightly
contract: positional `(sampled_token_ids, num_tokens_no_spec, token_ids_cpu)`,
**no req_ids** — rows identified by a 32-token content key; the pinned-era
`(num_spec, input_batch, …)` contract is auto-detected). 8 prompts × 48
tokens, greedy, Qwen2.5-0.5B bf16 eager.

| Gate | Status | Evidence |
|---|---|---|
| G2a (verified-by-construction) | **PASS** | 8/8 token streams byte-identical drafter-on vs drafter-off; 0 tie-flips, 0 divergences. Across all four engine boots of the day: **32/32 prompt-runs identical**, including poisoned and warm-started stores. |
| G2b (adversarial store) | **PASS** | store pre-poisoned with 200 garbage draft records → still 8/8 identical; only throughput noise. The Byzantine-tolerance claim is now measured, not argued. |
| G2c (persistence half) | **PASS** | after run 1 the store held 200 poison + **8 real sealed drafts** (atexit flush); a fresh engine process warm-started from them, stayed 8/8 identical, sealed 8 more (216 total). Cross-process draft persistence works end-to-end. |
| G2c (throughput) | first light only | drafter multiplier 4.4×–4.9× in cold-boot runs (15.6→68.5, 11.8→58.3 tok/s) but base itself swung 11.8→67.7 across boots (busy machine, no control gate). Magnitude awaits the Hoefler harness on an idle box; the workload is also favorably self-similar. |

New traps (T9, T10): **never place the κ-store under /tmp on WSL** — the VM
silently restarts and tmpfs evidence vanishes (one persistence run had to be
redone on a home-dir store); the nightly `custom_class` propose contract
passes numpy arrays — `if not sampled_ids:` is an ambiguity crash, guard with
`len()`.
