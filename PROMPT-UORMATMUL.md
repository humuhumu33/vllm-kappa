# PROMPT-UORMATMUL — uor-matmul in vLLM for deterministic, faster CPU decode

You are an engineering agent. Goal: integrate `UOR-Foundation/uor-matmul`
into the vLLM CPU inference path on THIS machine (Ryzen AI MAX 390 / Strix
Halo, no CUDA, WSL2, the κ-connector vLLM lane already live serving
Qwen2.5-0.5B-Instruct), and answer two measured questions:

1. **Determinism** — can uor-matmul make CPU decode produce *byte-identical*
   logits regardless of batch shape / reduction schedule (closing the G0b
   m-boundary variance that this whole κ program has been deferring to a
   GPU)?
2. **Throughput** — can it raise tok/s over the current backend, or not?

Benchmark model is the one already serving: **Qwen2.5-0.5B-Instruct, bf16,
CPU**. Discipline is the house standard (vllm-kappa MEASUREMENT-LOG): a
machine-load control on every perf rep, isolated boots, counted byte
identity, and **negatives published with the same prominence as wins**.

## 0. What uor-matmul actually is (studied 2026-08-25)

A Rust GEMM library whose thesis is *"decode the code, accumulate exactly,
encode once."* One method, many factorizations, all asserted to produce the
**same bytes** by its `CD-*` conformance gates — across tier, reduction
schedule, and substrate. Surface:

- `slice::gemm(m,k,n, a:&[i8], b:&[i8], c:&mut[i32], scratch)` — exact
  integer accumulation, the flagship path.
- `slice::gemm_float` / `gemm_float_ex` — f32/f64, `C := αAB + βC`, leading
  dims.
- `raw::sgemm` / `raw::dgemm` — **signature-identical to the `matrixmultiply`
  crate** (the drop-in seam).
- Never allocates (caller passes scratch; `suggested_scratch(shape)`).
- Views take arbitrary strides (transpose is a stride, not a mode).

Prior ground truth you MUST respect, not rediscover (memory
`holo-uor-matmul-validated`, `holo-vllm-uor-negative`):
- Replicated on Zen 5: **i32 path ~164× vs a naive ndarray triple-loop** —
  but that is vs *naive*, NOT vs a tuned BLAS.
- **Measured NEGATIVE (2026-08-21): uor-matmul is CPU-only and ~3.5× behind
  oneDNN** on the float GEMM that vLLM's CPU backend actually calls. A naive
  "swap the GEMM and watch tok/s rise" WILL LOSE. This prompt exists to find
  the framing where it does not.
- Two known defects logged in the validation memo — check them before
  trusting any edge shape.

## 1. The honest thesis — determinism is the product, speed is conditional

vLLM's CPU GEMM is oneDNN/PyTorch-CPU: SIMD-tuned, fast, and **schedule-
nondeterministic** (tiling/threading reorders float reductions → the G0b
byte-variance). uor-matmul is slower at f32 but **exactly schedule-
invariant**. So:

- **Determinism is a WIN uor-matmul can actually deliver on f32 today** —
  and it is the exact thing the κ program said needed a GPU's
  `VLLM_BATCH_INVARIANT`. If uor-matmul gives regime-free canonical bytes on
  a CPU, that alone justifies the integration (it makes G2a/G0b/G1b byte-
  identity hold on this machine, unlocking the publishable-on-CPU claims).
- **Throughput is only a win in two specific framings**, each to be measured,
  not assumed:
  1. **The integer path.** If the model is int8-quantized, `slice::gemm`
     (i8→i32 exact) competes on a very different curve than f32-vs-oneDNN,
     AND is deterministic by construction. This is the most promising tok/s
     lever — pair determinism with quantization so the comparison is
     int-exact-vs-int, not float-vs-BLAS.
  2. **The memoization ladder** (matmul-collapse thesis, `holo-matmul-
     collapse`): a coded operand whose `d(c_i)` is content-addressed is a
     retrieval, not a recompute — the hologram 157ns memo law at tile
     granularity. Speculative (L2+); gate behind a recurrence census before
     building any mechanism.

## 2. The seam — minimal-diff ladder (file < env < patch < fork)

vLLM's CPU GEMM is not a plain callable you can monkeypatch from Python; it
is inside compiled kernels / torch. So the integration ladder is:

1. **torch custom-op (no vLLM fork).** Build a `uor_matmul` PyO3/cffi
   extension exposing `sgemm`/`i8gemm`; register it as a torch custom op;
   route ONE hot linear (start with the LM head or the MLP down-proj) through
   it via a tensor-subclass or a `torch.library` impl. Measure there before
   widening. This is the smallest real diff.
2. **Numpy/ctypes harness first (pre-integration truth).** Before any torch
   binding, bench uor-matmul vs numpy(BLAS) vs the exact preimage shapes
   Qwen2.5-0.5B decode actually issues (below) — kill or confirm the thesis
   in an afternoon, no vLLM changes at all. **Do this first.**
3. Only if 1–2 justify it: a vLLM CPU-backend patch behind a pinned SHA.

## 3. Gates (each emits one JSON line; controls mandatory)

| Gate | Question | Method | Green / Kill |
|---|---|---|---|
| **M0 shape census** | What GEMMs does 0.5B decode actually issue? | instrument the running engine (or read the model config): per-layer (m,k,n), dtype, count per token | the shape table — the input to every later gate |
| **M1 raw race** | uor-matmul vs numpy-BLAS vs naive, on M0's real shapes | `crates`-built bench + numpy, f32 and i8, 5-rep medians + load control | publish the ratio curve; expect f32 LOSS (~3.5×), watch the i8 curve |
| **M2 determinism** | Is uor-matmul byte-identical across schedules where BLAS is not? | same GEMM, vary thread count / tiling / batch-m; hash the output bytes | uor-matmul: 1 unique hash across all schedules; BLAS: >1 ⇒ the determinism win is real and demonstrated |
| **M3 one-op integration** | Route one Qwen linear through uor-matmul via torch custom op | correctness (logits match reference within tol) + tok/s delta + determinism of that op's output | correctness holds; report tok/s honestly; determinism of the routed op proven |
| **M4 end-to-end determinism** | With the hot ops routed, do full decode logits go byte-identical across batch shapes? | G2a-style: same prompt, batch 1 vs 16, count identical tokens at 48 & 128 | any increase in byte-identical fraction over the oneDNN baseline is the headline; 100% = G0b closed on CPU |
| **M5 int8 throughput** | Quantize 0.5B to int8; race i8 uor-matmul decode vs bf16 oneDNN decode | end-to-end tok/s + quality delta + determinism | tok/s ≥ bf16 baseline AND deterministic ⇒ the double win; else publish the honest trade |

Kill criteria, stated up front so the negative is a result, not a
disappointment: if M1 f32 loses (expected) AND M5 int8 does not recover it,
then **uor-matmul's contribution to vLLM-on-CPU is determinism, not speed** —
which is still a genuine, publishable unlock (regime-free canonical bytes on
CPU) and exactly what the κ verification program needs. Say so plainly.

## 4. Non-negotiables
- Bench against the model already serving (Qwen2.5-0.5B-Instruct); do not
  download a new one (disk is at ~5 GB free).
- uor-matmul is a pinned submodule/crate; build it, never vendor-edit it.
- Determinism claims are byte-level and counted; throughput claims carry the
  machine-load control; every uor-matmul result cross-checks its own CD-gate
  bytes.
- Respect the two known defects; add any new trap to this repo's log.
- Append findings to MEASUREMENT-LOG.md in the dated, retraction-honest
  style. If the answer is "determinism yes, speed no," that sentence is the
  deliverable.
