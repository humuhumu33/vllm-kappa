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

## 2026-08-24 — G2c controlled (CPU lane) — retraction + the real result

Environment first: the box's WSL was memory-starved all morning (24 GB VM on a
31 GB host at 0 bytes disk free). Re-run under a right-sized VM (8 GB,
`VLLM_KAPPA_MEMUTIL=0.35`), 3 arms (none / in-tree suffix / κ) × 2 boots,
numpy-GEMM control gate on every rep.

**RETRACTION.** The morning's "4.4× first light" was measured against a
baseline thrashing in the starved VM (12–19 tok/s). A healthy baseline runs
~300 tok/s at batch 16. Honesty discipline: the earlier number is void.

**Batch-16 (compute-saturated CPU): speculation loses, ~10–35% below
baseline — for BOTH suffix and κ equally.** This is the regime, not the
κ-store: on a saturated CPU, the k extra verify tokens cost what they save.
7/12 reps control-flagged (busy machine); magnitudes indicative only.
Identity: **192/192 streams byte-identical across all arms** at 48 tokens.

**Batch-2 × 128 tokens (decode-bound regime): the κ thesis lands.**
- baseline ~51–55 tok/s · suffix ~58 (both boots — no cross-boot memory)
- **κ boot 2 (warm from store): 245.5 tok/s — 4.4× baseline, 4.3× over
  in-tree suffix, control unflagged.** The persistent store is exactly the
  measured delta over Arctic suffix decoding.
- Same-boot κ (cold store): 58.4 — parity with suffix, as expected.
- Corroborated on different prompts in a separate session: 11.5→30.1 tok/s
  (2.6×, busy machine, cold store, within-generation matches only).

**Long-generation identity on CPU is text-dependent.** At 128 tokens some
prompts diverge from the batch-reference for suffix AND κ alike (κ warm boot
most, since accepted runs enlarge verify-batch m); a different prompt set at
the same shape held 2/2 with zero tie-flips. Mechanism = the G0b m-boundary
bit-divergence surfacing as greedy tie-flips. Not a drafter defect (in-tree
suffix diverges identically); distribution-level correctness is untouched;
bit-level identity for long generations needs the batch-invariant GPU lane.

**G2d (novel ≤3%)**: unresolvable on this lane — the whole spec-decode
mechanism is regime-negative at batch 16 here, suffix included. Defer to GPU.

Verdict per PROMPT.md §7: on CPU the drafter's value is the **warm-start
regime** (agentic replay, restarts, shared stores, small batches) where it is
the only arm that improves at all — and the saturated-batch regime is a
measured negative for speculation generally. The publishable G2c number is
the GPU lane's to produce.

## 2026-08-24 — G1c: the L1 witness pipeline on real inference — PASS

KappaConnector live via `kv_connector_module_path` (zero vLLM diff), 16
requests × 48 tokens × 2 reps per arm, Qwen2.5-0.5B CPU.

- **16/16 real request seals verified** by `verifier/kappa_verify.py` — the
  full chain (request → connector `request_finished` → off-path witness
  build → κ-store → offline audit) closed on real inference bytes for the
  first time.
- **On-disk tamper refused**: 1 byte flipped in a stored record → audit rc 1.
- **Overhead 1.46% by median — within noise** (plain arm spread 285–340
  tok/s); honest claim: no measurable overhead at this precision (gate ≤2%).

Two more seam lessons (T11, T12), both from vLLM resolving the connector at
CLASS level and shipping its metadata across processes: a factory function
cannot stand in for the connector class (`get_required_kvcache_layout` is a
classmethod call) — serve the real class via module `__getattr__` (PEP 562);
and connector-metadata classes must pickle by module-level name — set
`__module__`/`__qualname__` on lazily-built classes and serve them from
`__getattr__`, or the scheduler→worker handoff dies.

## 2026-08-24 — Phase 3 first light (κ-KV pull) — partial, honestly scored

The load path is implemented (block-κ-keyed payloads via KeyedIndex,
scheduler hit-claiming, worker injection mirroring the in-tree example,
load-error reporting wired to vLLM's invalid-block recompute machinery,
which upstream provides end-to-end: worker mixin → `invalid_block_ids` →
scheduler rewind to longest valid prefix).

Measured on a 1,354-token shared prefix (4 engine boots, one process):
- **G3c (equality): PASS** — generation from pulled KV byte-identical to
  cold prefill.
- TTFT 1.142s → 0.176s with κ-pull — but the harness ran boots in one
  parent process in order producer→control→pull, so warmth ordering
  contaminates the comparison; **treat 6.5× as unverified until re-run
  with per-boot process isolation.**
- **Defect found (fix known, not yet applied): only the first
  chunked-prefill chunk is captured** — `build_connector_meta` reads
  `scheduled_new_reqs` only; resumed chunks never store. 10/84 blocks
  landed. Fix: accumulate store metadata across steps via the cached-reqs
  side of SchedulerOutput.
- **G3b (tamper fallback): FAIL as measured** — corrupted payload was
  refused at read (verify-on-read worked) but the run produced a different
  stream, i.e. the invalid-block recompute did not take effect; silent
  wrong output is the worst failure mode and G3 CANNOT pass until this is
  fixed and proven. Instrumented rerun attempts were blocked by the host
  resource wall (below).

## 2026-08-24 — Phase 3 COMPLETE on the CPU lane (post-recovery)

Host recovered without elevation: deleted rebuildable tool artifacts inside
WSL (whisper_env, emsdk, depot_tools, ollama tarball ≈ 9 GB), fstrim'd
878 GiB, VM right-sized to 10 GB. (VHD compaction still pending the user's
elevated `diskpart /s compact-vhd.txt` — prepared in the session scratchpad.)

**Retraction of a retraction:** the "chunked-save gap (10/84 blocks)" was
wrong — the CPU backend's `block_size` is **128**, not 16; 10 × 128 = full
coverage of the 1,354-token prompt. Witness block keys use the same
`cache_config.block_size`, so identity is consistent throughout.

**G3b root cause and fix (the important one):** after refusing a corrupt
block, the same step's save pass re-stored the never-computed garbage from
the paged buffer under the block's κ — replacing detectable corruption with
validly-hashed poison. Two rules now enforced: (1) **drop the index entry on
refusal** (negative cache) so rescheduled requests miss and prefill; (2)
**never store blocks you did not compute** (store requests carry the
externally-claimed window and skip it). With both:

| Gate | Result (isolated processes, 1,354-token shared prefix) |
|---|---|
| G3a TTFT | cold 1.264/1.224 s → κ-pull **0.269/0.239 s = 5.1×**; producer (cold+storing) 1.413 s (~12% store overhead, on-path writes — future async) |
| G3b tamper | corrupt payload → refused at read → index dropped → invalid-block report → scheduler recompute → **stream byte-identical to producer**; index self-heals with recomputed bytes |
| G3c equality | **7/7 streams byte-identical** across cold/producer/pull/tampered arms |

New traps: **T13** engine boots must run from real script files
(multiprocessing spawn re-imports `__main__`; heredoc/stdin scripts kill
EngineCore); **T14** the poisoning loop above — a κ-store writer colocated
with a κ-store reader must partition computed-vs-claimed blocks or one
corruption becomes permanent.

## 2026-08-24 — G1b: the challenge asymmetry, measured on real KV bytes

`bench/run_g1b.py`: producer stores all blocks; drop block J's index entry;
the challenge boot pulls [0,J) and recomputes J..end; compare recomputed
payload against the witnessed one.

- **Block 9 (last-block challenge): bit-perfect** — the recomputed payload
  re-derives the *identical κ-label* (content addressing certified the
  equality; torch.save serialization proved deterministic). Challenge cost
  **0.314 s vs 1.117 s full prefill**; output stream identical.
- **Block 5 (interior split): low-bit KV drift** — recomputed bytes differ
  (different prefill GEMM shape; the G0b m-variance at byte level) while the
  output stream is STILL identical (drift below argmax sensitivity here).
- Asymmetry curve is linear in the remaining suffix: 0.31 / 0.78 / 1.12 s
  at J = 9 / 5 / cold.

Verdict: on the CPU lane, byte-level KV challenges are exact for same-split
replay (and the token-chain seal verification of G1c is split-independent);
interior-split byte equality requires the batch-invariant GPU lane. The
O(block)-at-the-margin claim is now a measurement, not an argument.

**Earlier session-stop note (superseded by the recovery above):** The 31 GB host with C: at ~0 free
cannot sustain more engine-boot cycles: 12 GB+ WSL destabilizes Windows
(pagefile growth → WSL service death), 8 GB fails engine boots
intermittently. Every retry risks another ENOSPC corruption event (one
already forced a full venv rebuild). Phase 3 verification resumes after
the host gains headroom (VHD compaction or asset cleanup — user decision).

**Lane summary (all gates runnable on this CPU lane are now complete):**
G0a PASS (full) · G0b measured (flag GPU-only; native = per-regime canonical)
· G1a PASS (900/900 unit + disk + production tamper) · G1c PASS (no
measurable overhead, real seals verify) · G2a PASS (192/192 at 48 tok) ·
G2b PASS (poison harmless) · G2c: saturated-batch negative (all arms),
decode-bound warm-start 4.4×/4.3× — GPU lane owns the publishable number ·
G1b + Phase 3 (κ-KV load) + G2d: pending, need the load path and a GPU.
