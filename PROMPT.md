# vllm-kappa — implementation prompt

You are implementing **the κ-connector**: a zero-fork extension to vLLM that makes
inference **100% κ-address verifiable** and **faster on any workload with repetition**,
by giving values, sharing, and verification to the κ-addresses vLLM already mints.
This document is the complete specification. Execute it as a measured research
program: every phase has gates, every claim gets a number, negative results are
committed, and the diff against upstream vLLM is **zero**.

---

## 0. Mission and the one inversion that makes it work

Modern inference is unverifiable (you trust whoever ran the GPU) and amnesiac
(identical work is recomputed forever). The UOR doctrine — *the name of an answer is
its proof; anything proven once is never computed again* — fixes both, but only at
granularities where repetition actually exists and only over deterministic bytes.

The design exploits one rare property: **speculative decoding is the only protocol in
computing where verifying untrusted work makes you faster.** The target model verifies
k drafted tokens in one forward pass; rejection sampling guarantees the output
distribution is exactly the target model's. A wrong, stale, or malicious draft source
can only waste a speculation round — it can never corrupt one output token. Therefore
a shared, content-addressed, *untrusted* store of previously-verified continuations is
a Byzantine-tolerant accelerator with zero hot-path cryptography.

Three layers, all through public vLLM plugin seams:

- **L0 — canonical bytes**: `VLLM_BATCH_INVARIANT=1`. Same request ⇒ bit-identical
  logits and KV bytes regardless of batch composition. This is what makes κ(request)
  → output a re-derivable fact rather than a per-run accident.
- **L1 — κ-witness** (verifiability): a KV-connector plugin seals every KV block and
  the token stream into a κ-chain. Any block is verifiable by recomputing it from its
  predecessors: **O(one block), not O(prefix)**.
- **L2 — κ-drafter** (performance): a speculative proposer backed by the κ-store.
  Cross-request, cross-restart, cross-machine draft reuse, verified by construction.
- **L3 — κ-KV fabric** (performance): prefill KV blocks shared between instances by
  κ, spot-verified by the L1 recompute check. Prefill is paid once network-wide.

"100%" means: every output byte is bound into the κ-chain and any byte is
challengeable at O(block) cost by anyone holding the model. It does not mean every
byte is checked for free — full certainty is full recompute. Never claim otherwise;
zkML (10⁴–10⁶× overhead) is explicitly out of scope because it forfeits the
performance half of the mission.

---

## 1. Prior measured facts — do not re-litigate these

Established 2026-07-26 → 2026-08-24 on a Ryzen AI MAX 390 (Zen 5) and recorded in the
Hologram research logs:

1. **vLLM already mints UOR addresses.** `uor_addr.kappa.cbor_address(cbor2.dumps(k,
   canonical=True))` is **byte-identical** to vLLM's `sha256_cbor` prefix-cache block
   hash (`vllm/utils/hashing.py`). The address space exists; this project adds the
   values, witnesses, and network behind it.
2. **Kernel replacement is dead.** uor-matmul exact-i8 is 3.5× behind oneDNN
   `_int_mm`, exact f32 is 50–140× behind faer/OpenBLAS, and no GPU lane exists.
   Do not touch any GEMM. (Freivalds spot-checks of existing int8 GEMMs remain a
   legitimate optional extra — see §8.)
3. **Matmul/activation memoization is dead in serving**: activations are unique per
   token; hit rate ≈ 0. Token continuations and KV blocks are the only two
   granularities where serving-time repetition exists. That is where this design sits.
4. **uor-addr performance envelope** (rev `165b51e3`, the same rev hologram-fabric
   pins): ~4 µs fixed per call, ~730 MB/s steady, **no streaming API** (full buffer
   required), GIL released. Budget hashing accordingly (§6, Trap T3).
5. **`witness.verify()` is O(1) and never reads the payload** — it replays the
   pipeline trace. It proves an address was derived correctly; it does **not** prove
   bytes match a label. Byte integrity is always a fresh BLAKE3/derivation over the
   received bytes. Never conflate the two in code or prose.

## 2. Pinned substrate — clone, never fork

| Component | Repo | Pin |
|---|---|---|
| vLLM | `vllm-project/vllm` | `1baf372bf7c05a05ed4028f42d7d15b8d908a95a` (update deliberately; re-run every gate on any bump) |
| uor-addr | `UOR-Foundation/uor-addr` | `165b51e3` (hologram-fabric's pin) |
| hologram-fabric | `Hologram-Technologies/hologram-fabric` (private) | current main |
| hologram-storage | `Hologram-Technologies/hologram-storage` | fabric's pin `ae71f47c` |
| kappa-registry | `UOR-Foundation/kappa-registry` | fabric's pin `2929832c` |

**vLLM is cloned and pip-installed unmodified.** All extension code lives in this
repo's `vllm_kappa` Python package and rides four public seams (verified present at
the pinned SHA):

| Seam | Where | Used for |
|---|---|---|
| Batch-invariant mode | `VLLM_BATCH_INVARIANT` (`vllm/envs.py:629`), kernels in `vllm/model_executor/layers/batch_invariant.py` | L0 |
| KV-connector plugin API | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` + factory registration (out-of-tree connectors register by class path, as LMCache does) | L1, L3 |
| Custom speculative proposer | `vllm/v1/spec_decode/custom_class_proposer.py` — loads any class by module path from `speculative_config.model="vllm_kappa.drafter.KappaProposer"`; must expose `propose` (and a no-op `load_model`) | L2 |
| Suffix decoding reference | `vllm/v1/spec_decode/suffix_decoding.py` (Arctic Inference `SuffixDecodingCache`) | L2 — reuse its tree logic; replace its per-instance, process-lifetime cache with the κ-store |

**Reuse rule:** before writing any component, check whether vLLM or a pinned Hologram
library already contains it. The drafter reuses Arctic's suffix tree; the connector
imitates the structure of `lmcache_connector.py`; all storage, addressing, transport,
and auth come from hologram-fabric. This repo should end up **small** — target ≤ 1,500
lines of non-test Python. If a file grows past 400 lines, you are probably rebuilding
something that exists; stop and look again.

## 3. hologram-fabric is the entire κ backend — use it, build none of it

Fabric ("the modular HTTP composition layer for the UOR and Hologram architecture")
exposes the real Rust libraries behind stable authenticated HTTP. The plugin package
therefore contains **no hashing protocol, no storage engine, no networking, no auth**:

| Fabric surface | Endpoint | vllm-kappa use |
|---|---|---|
| Hologram Storage | `POST/GET /api/v1/hologram/storage/objects[/blake3:<digest>]` | the κ-store: witness records, draft continuations, KV block payloads (L3). Content address **is** the integrity check on read. |
| UOR Address | `POST /api/v1/uor/addr/json/sha256` | canonical address derivation + replay-witness verification for witness records; cross-check that plugin-side addresses match fabric-side |
| Kappa Registry | `/v2` (standard root) | distribution of large sealed artifacts (witness bundles, exported KV segments) between sites |
| Auth | `Authorization: Bearer` / `X-Hologram-Token`; capabilities `hologram.storage.write`, `uor.addr.encode`, `kappa.registry.access` | one token per vLLM instance; scope minimally |

Client rules:
- One thin async client (`vllm_kappa/fabric.py`, httpx): `put(bytes) -> κ`,
  `get(κ) -> bytes` (re-derive and compare on read — trust nothing), `derive(json)`,
  batched where fabric allows.
- **Two-tier store**: local disk/RAM cache in front (hot lookups are µs and offline-
  safe), fabric behind (shared truth). The drafter reads the local tier on the hot
  path *only*; fabric sync is a background task.
- **Fail-open always**: fabric unreachable ⇒ behavior degrades to vanilla vLLM
  (drafter proposes nothing; connector skips witnessing with a counter incremented).
  A network store must never be able to stall or crash serving.
- If a needed capability (e.g. batch object put, range listing) is missing from
  fabric, file the requirement against hologram-fabric — do not grow a parallel
  protocol in this repo.

## 4. Non-negotiable invariants

1. **Zero vLLM diff.** `git -C third_party/vllm diff` is empty forever. CI asserts it.
2. **Output-distribution equality.** With the drafter enabled and greedy sampling, the
   token stream is byte-identical to drafter-off for every prompt. This is both a
   correctness gate and the headline demo.
3. **The hot path never blocks on the network.** All fabric I/O is async and off-path;
   drafter lookups have a hard local-tier deadline (≤ 200 µs, measured not assumed);
   witnessing is queued off-path.
4. **κ is the only identity.** No UUIDs, no hostnames-as-identity, no second naming
   surface anywhere in the package.
5. **Determinism before verifiability.** No witness is emitted unless
   `VLLM_BATCH_INVARIANT=1` was verified on this hardware (Phase 0 gate); a witness
   over nondeterministic bytes is a lie. The drafter (L2) works without invariance;
   L1/L3 refuse.
6. **Honesty discipline** (house rule from uor-vv / the uor-matmul benchmark fork):
   report confidence intervals, commit rejected runs, concede what loses. The
   determinism tax is measured and printed next to every win.

## 5. Repository layout (deliverable)

```
vllm-kappa/
  PROMPT.md                  # this file
  README.md                  # visual-first, measured charts only, no house jargon
  THREAT-MODEL.md            # adversarial self-audit (poisoned store, tampered KV,
                             #   replayed witness, wrong-model witness, DoS via store)
  REPLICATE.md               # <30 min replication path, pinned SHAs, exact commands
  MEASUREMENT-LOG.md         # every run, including rejected ones
  vllm_kappa/
    __init__.py              # version + seam-compat assertions against pinned vLLM
    addressing.py            # κ derivation glue: reuse vLLM sha256_cbor identity fact;
                             #   BLAKE3 for byte digests; NO new formats
    fabric.py                # thin async fabric client (two-tier store, fail-open)
    witness.py               # L1: witness record schema (CBOR, canonical), chain builder
    connector.py             # L1+L3: KVConnector (structure mirrors lmcache_connector)
    drafter.py               # L2: KappaProposer (wraps Arctic suffix tree over κ-store)
  verifier/
    kappa_verify.py          # standalone CLI: replay/spot-check a witness chain
                             #   against any vLLM instance; O(block) challenge
  bench/
    workloads/               # agentic-replay, RAG-shared-prefix, ShareGPT, novel-text
    run_bench.py             # Hoefler-rules harness; control gate (idle machine);
                             #   emits per-phase JSON + charts
  tests/                     # unit + integration (vLLM in-process, small model)
  third_party/vllm           # clone @ pinned SHA (submodule or vendored; UNMODIFIED)
  .github/workflows/ci.yml   # lint, tests, zero-diff assert, gate summaries
```

## 6. Phases and gates

Model for all phases: **Qwen2.5-1.5B-Instruct** (small, GQA, ubiquitous) on GPU when
available; the CPU backend is an accepted fallback for every gate except throughput
absolutes. Block size = vLLM default 16. Every gate produces a JSON artifact in
`MEASUREMENT-LOG.md`.

### Phase 0 — substrate proof (½ day)
- Pin + build vLLM; smoke-serve.
- **Gate G0a (identity):** re-verify `sha256_cbor(block key) ≡ uor-addr
  cbor_address` byte-for-byte at the pinned revs, in CI, as a unit test. This
  equality is the project's foundation stone; if a vLLM bump ever breaks it, the bump
  is rejected.
- **Gate G0b (determinism):** with `VLLM_BATCH_INVARIANT=1`, 100 prompts × 5 batch
  compositions × 2 process restarts ⇒ bit-identical logits and (sampled) KV bytes.
  Record the invariance tax: throughput invariant-on vs off on the bench workloads.
  If invariance does not hold on this hardware/model, L1/L3 are **blocked** — say so
  in the log and continue with L2 only.

### Phase 1 — κ-witness chain (L1; ~2 days)
- `connector.py`: on block commit, enqueue witness = canonical-CBOR record
  {κ(block-key), BLAKE3(KV bytes), parent κ, model-root κ, engine fingerprint
  (vLLM SHA, dtype, TP layout), sampled token ids}. Chain per request; seal final
  record to κ(request). Push to fabric storage off-path.
- `verifier/kappa_verify.py`: given a witness chain and any vLLM instance with the
  model — (a) integrity mode: re-derive every address over supplied bytes; (b)
  challenge mode: pick random block i, reconstruct its 16-token chunk-prefill from
  blocks < i, compare KV bytes byte-identical.
- **Gate G1a (tamper):** flip 1 byte in any KV block or any witness field ⇒ refusal,
  100/100 trials.
- **Gate G1b (asymmetry):** measured challenge cost is O(block) and flat in prefix
  length (plot 1k/4k/16k-token prefixes).
- **Gate G1c (overhead):** end-to-end throughput with witnessing on ≤ 2% below
  invariant-mode baseline. Trap T3 arithmetic first: per-block KV for the target
  model ≈ 2·layers·16·kv_heads·head_dim·dtype bytes (≈ 0.9 MB for a 7B-class GQA
  model) ⇒ hash off-path, multithreaded BLAKE3, and only for blocks that are already
  leaving the GPU (offload/share) or explicitly requested; token-level witnesses are
  always on (they are tiny).

### Phase 2 — κ-drafter (L2; ~2 days) — the headline
- `drafter.py`: `KappaProposer(vllm_config)` exposing `propose(...)` with the same
  batch contract as `suffix_decoding.py`. Internals: Arctic `SuffixDecodingCache`
  for the in-process tree, extended with (a) persistence: accepted continuations
  sealed to κ(context-suffix window) and pushed to the two-tier store; (b) warm
  start: on boot and periodically, merge store entries into the tree; (c) provenance
  counter: drafts served from local-novel vs store-recalled.
- Wire-up is configuration only:
  `--speculative-config '{"method":"custom", "model":"vllm_kappa.drafter.KappaProposer", "num_speculative_tokens":8}'`.
- **Gate G2a (verified-by-construction):** greedy, 500 diverse prompts: token streams
  byte-identical drafter-on vs drafter-off. This gate IS the security argument.
- **Gate G2b (adversary):** poison the store with wrong/garbage continuations ⇒ G2a
  still passes; only acceptance rate and latency move. Record the worst-case slowdown
  cap (bounded by num_speculative_tokens per round).
- **Gate G2c (performance):** on agentic-replay and RAG workloads, decode throughput
  vs (i) no spec decode, (ii) in-tree suffix decoding. Win condition: beat (i) by
  ≥ 1.5× and beat (ii) whenever a second instance or a restart is involved (the
  shared/persistent store is the entire delta vs Arctic — demonstrate exactly that:
  kill the process, restart, first request drafts from the store immediately).
- **Gate G2d (do-no-harm):** on fully-novel text, throughput within 3% of
  no-spec-decode baseline (lookup misses must be ~free).

### Phase 3 — κ-KV fabric (L3; ~2-3 days)
- Extend `connector.py`: instance A pushes witnessed prefill blocks (κ-keyed) to
  fabric storage; instance B, on prefix-cache miss, queries by block-key κ, pulls,
  **spot-verifies r random blocks via the G1 challenge**, injects, serves.
- **Gate G3a (TTFT):** shared 8k-token prefix, cold instance B: TTFT with κ-pull vs
  local prefill. Report the crossover prefix length under the measured
  link bandwidth (KV bytes are big — state the honest break-even, don't hide it).
- **Gate G3b (tamper):** corrupted block from the store ⇒ spot-check refuses,
  instance falls back to local prefill, request still succeeds.
- **Gate G3c (equality):** tokens generated from pulled-KV ≡ locally-prefilled KV,
  byte-identical (requires G0b).

### Phase 4 — report (~1 day)
- README with the three measured charts: (1) verify-cost asymmetry (O(block) flat
  line vs O(prefix) recompute), (2) drafter throughput by workload incl. the
  determinism tax shown honestly as a negative bar, (3) TTFT vs prefix length with
  the crossover marked. THREAT-MODEL.md and REPLICATE.md per the house pattern
  (humuhumu33/uor-matmul `benchmark` branch is the calibration reference for tone:
  audience-calibrated, no internal jargon, concessions first).

## 7. Kill criteria — commit the corpse, don't bury it

- G0b fails on target hardware ⇒ L1/L3 are re-scoped to "per-deployment
  reproducibility" (same engine instance only) and the README says so plainly.
- G2c wins < 1.2× on every workload ⇒ the drafter is not worth its complexity;
  publish the negative with the acceptance-rate data and stop at L1.
- G3a crossover prefix length > 16k tokens on a 10 Gb link ⇒ κ-KV is a niche
  (cross-site only); scope it to that and say so.

## 8. Optional extra (only after Phase 4 gates pass)
Freivalds sidecar: for W8A8-quantized deployments, spot-check sampled linear-layer
GEMMs (exact-i32 accumulation licenses the check; measured 5×/9×/14× cheaper than
recompute at n = 1024/2048/4096). Separate module, separate flag, never on the
default path.

## 9. Traps (all previously hit — do not rediscover them)

- **T1** `witness.verify()` never reads the payload (fact 5 in §1). Integrity =
  fresh derivation over received bytes, every read, no exceptions.
- **T2** uor-addr has no streaming API and ~4 µs fixed cost per call — batch small
  records into canonical-CBOR bundles before addressing; never address per-token.
- **T3** KV-byte hashing budget (see G1c): ~0.9 MB/block for 7B-class GQA. Off-path,
  multithreaded BLAKE3, piggyback on transfers that happen anyway.
- **T4** The KV-connector v1 API surface moves between vLLM releases — the
  `__init__.py` seam-compat assertions must check the exact classes/signatures at
  import time and fail loudly with the pinned-SHA message.
- **T5** `custom_class_proposer` instantiates with `vllm_config` only and requires a
  callable `propose`; suffix-decoding's `propose` batch contract (see its
  `input_batch` handling) is the contract to match.
- **T6** Sampling params are part of the witness identity — κ(request) must cover
  seed, temperature, top-p, and the engine fingerprint, or two honest instances will
  "refute" each other.
- **T7** Windows dev box: kappa-registry/fabric store on NTFS hits the colon trap —
  run fabric under WSL with the store on ext4.
- **T8** Never benchmark on a busy machine: the control-gate pattern (reject the
  session if the cubic control drifts) is mandatory for every published number.

## 10. Definition of done

A stranger with one GPU, the pinned SHAs, and REPLICATE.md can in under 30 minutes:
(1) serve a model where every response carries a κ-sealed witness chain; (2) tamper
with one KV byte and watch the verifier refuse it, at measured O(block) cost; (3)
restart the server and watch the first request draft from the persistent κ-store;
(4) poison the store and watch outputs stay byte-identical while only latency moves;
(5) read honest charts where the determinism tax is as visible as the wins — with
`git diff` in the vLLM clone empty the whole time.
