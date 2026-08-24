# PROMPT — vLLM on the UOR Universal Lossless Encoding Standard

You are an engineering agent. Your mission: bring the **entire vLLM inference
engine** onto the UOR content-addressing standard with the **smallest possible
change**, then **measure and validate** every notable advantage — verification,
latency, throughput — with numbers that survive a hostile reviewer.

This is a measurement program, not a rewrite. Elegance here means: find the
one rule vLLM already almost follows, make it total, and prove what that buys.

---

## 0. Study set (read before writing any code)

| Source | What it contributes |
|---|---|
| `crates.io/crates/uor-foundation` (v0.5+) | The ontology. Sealed pipeline `host bytes → Grounding → Datum → Validated → Grounded → certificate()`; resolvers return `Result<Certified, Witness>`. **Adopt its semantics** (certify-or-witness, refusal as a first-class outcome), not its Rust machinery. Conformance is binary — no shims. |
| `github.com/UOR-Foundation/uor-addr` | The encoding rule. An address is the hash of a **published canonical form** — "two values that mean the same thing get the same identifier." Format `sha256:<64hex>`. The CBOR realization (`kappa.cbor_address`) = sha256 over canonical CBOR. |
| `github.com/UOR-Foundation/kappa-registry` | The store discipline. BLAKE3 content addressing over raw blob bytes; anchors = sha256(dCBOR); tags as the *only* mutable naming tier; **verify-on-read: bytes that do not derive their label are refused**. |

**The UOR doctrine in one line:** *canonicalize losslessly → hash → that IS the
name; verification is re-derivation; a mismatch is a refusal, not an error.*

## 1. Ground truth already measured (do not rediscover)

The repo you are standing in (`humuhumu33/vllm-kappa`) has a MEASUREMENT-LOG
with pinned results on vLLM nightly (CPU lane, Qwen2.5-0.5B). Load-bearing:

- **The coincidence (G0a, byte-verified):** vLLM's opt-in
  `prefix_caching_hash_algo="sha256_cbor"` block hashing ≡
  `sha256(cbor2.dumps(obj, canonical=True))` ≡ uor-addr's `cbor_address`
  rule. vLLM already *almost* mints UOR addresses for every KV block.
- **The gap:** the **default** is `"sha256"` = sha256 over **pickle** — not a
  canonical form, not UOR. xxhash modes are non-crypto with process-random
  roots. And every vLLM identity is an **input-address** (token lineage), not
  a **byte-address** (content); nothing is ever verified against it.
- **Regimes (G0b, G1b):** CPU kernels are canonical **per batching regime**
  only (GEMM m-boundary bit-variance); byte-level equality claims need the
  batch-invariant GPU flag or same-split replay. Restart-stable either way.
- **Proven seams (zero vLLM diff):** KV-connector via
  `kv_connector_module_path` (witness seals: 16/16 real requests verified,
  overhead within noise, tamper refused); speculative drafts via
  `custom_class_proposer`; KV payloads per block-κ with cross-boot pull
  (**5.1× TTFT**), tamper → refuse → recompute → identical output.
- **Regime economics (G2c):** speculation is a measured **negative** on a
  saturated-batch CPU (vLLM's own suffix arm loses identically); the κ win
  regime is decode-bound + warm store (**4.4× / 4.3× over in-tree suffix**).
  Publishable throughput numbers belong to the GPU lane.
- **Traps T1–T14** (in MEASUREMENT-LOG): connector class-level resolution
  (PEP 562), cross-process metadata pickling, never store uncomputed blocks,
  drop index entries on refusal, never /tmp on WSL, `len()` on numpy, etc.

## 2. The design — three planes, two functions, one doctrine

Every UOR obligation in vLLM reduces to exactly two derivations:

- **identity(x)** = `sha256(canonical_cbor(x))` — for *meanings* (what was
  asked, of which model, under which sampling). This is uor-addr's rule and
  vLLM's existing `sha256_cbor` — the rule is already in-tree.
- **kappa(b)** = `blake3(b)` — for *bytes* (weights, KV payloads, seals,
  outputs). This is kappa-registry's rule.

Nothing else. Elegance = refusing to invent a third function.

### Plane I — Identity (make the existing rule total)
Flip the engine onto canonical identity derivation everywhere it mints one:
1. Default `prefix_caching_hash_algo` → `sha256_cbor` (config, not code).
2. Produce the **Identity Ledger**: enumerate *every* identity vLLM derives —
   block keys, mm hashes, LoRA extra keys, cache salts, prompt-embed hashes,
   engine/model fingerprint, request seals — with each preimage schema.
   For each: is the preimage canonically encodable? If not (pickle, repr,
   id(), random seed), that is a **finding**, and the fix is a schema, not a
   hack. The ledger is a deliverable — it is the conformance surface.
3. Gate **U1**: an *out-of-process* verifier re-derives 100% of minted
   identities from the ledger schemas using only the uor-addr rule.
   Anything unrederivable = not UOR = red.

### Plane II — Content (every byte artifact gets a κ)
Zero-fork ladder: env/config → plugin seam → surgical patch (in that order,
stop at the first rung that works):
1. **KV payloads** — done (this repo's connector). Keep.
2. **Model weights** — κ-address the weight artifacts at load: per-tensor (or
   content-defined-chunked) BLAKE3 labels, manifest = one identity-plane
   record binding model fingerprint → tensor κ list. Seam: vLLM's model
   loader is pluggable (`load_format` / loader registry) — no fork expected.
   Verify-at-load = recompute κ per tensor; refusal = engine will not boot on
   tampered weights. Prior art: holo.cpp measured verify ≈ load-time-only,
   ~0 decode cost — reproduce that shape of result here.
3. **Outputs** — request seals (done: witness chain) — ensure the seal binds
   Plane-I identity to Plane-II κ of any stored KV evidence.
4. **Drafts** — the κ-drafter store (done). Keep.

### Plane III — Witness (certify-or-refuse, ontology semantics)
Map `Result<Certified, Witness>` onto the toolchain: every verifier returns
either a certificate (all labels re-derive) or a witness naming the exact
object and byte offset that refused. Already half-built here
(`verifier/kappa_verify.py`); finish the semantics: exit codes, machine-
readable witness records, and **no third state** — "probably fine" does not
exist. Stretch (only if cheap): run the `uor-foundation` crate as an
*external reference verifier* over exported artifacts, making conformance a
property checked by the ontology's own code rather than ours.

## 3. What to measure (the entire point)

Rules of evidence — non-negotiable, learned the hard way:
- Every perf rep carries a **machine-load control** (the numpy-GEMM control
  gate in `bench/`); flagged reps are reported but never headline.
- TTFT/latency comparisons use **isolated cold boots** (fresh process per
  arm, alternating order). A warm-parent artifact retracted one 4.4× claim
  in this repo already; do not repeat it.
- Identity claims are **byte-level and counted** (n/n identical), with the
  regime stated (batch shape, invariance flag).
- Negative and regime-dependent results are published with the same
  prominence as wins. Retractions are logged, never overwritten.

| Gate | Question | Method | Green |
|---|---|---|---|
| **U1 identity totality** | Is every minted identity UOR-rederivable? | ledger + external verifier | 100% rederived |
| **U2 identity overhead** | Cost of canonical default vs pickle default vs xxhash? | A/B/A boots, batch sweep, per-block hash µs + end-to-end tok/s | ≤1% e2e; report the µs curve |
| **U3 weight verification** | Cost to boot a *proven* model? | verify-at-load on/off, cold + warm page cache | verify ≤15% of load, 0 decode cost; tamper = refusal |
| **U4 content dedup** | Do κ-labels deduplicate real artifact families? | κ-chunk a base model + ≥2 finetunes / adapters; measure shared-block %, bytes saved, second-model load time | any double-digit dedup is a headline; 0% is a publishable negative |
| **U5 latency (exists — re-verify)** | Cross-instance prefix pull & warm drafts | this repo's G3/G2c harnesses on the target lane | pull ≥2× TTFT at ≥1k-token prefix; drafter per its regime table |
| **U6 verification asymmetry** | Audit and challenge cost curves | full-audit cost vs serve cost; challenge cost vs suffix length (O(block) at the margin — measured 0.31s vs 1.12s here) | curves published; challenge ≪ recompute |
| **U7 refusal correctness** | Adversarial store, end to end | bit-flips in weights / KV / seals / drafts; poisoned stores | 100% refusal, 0 wrong bytes served, service degrades to recompute |

**Kill criteria** (publish the negative, keep the code that earned it):
U2 >3% overhead with no U3–U6 win ⇒ canonical-by-default is not free — say
so. U4 = 0% ⇒ content addressing buys integrity but not storage — say so.
All regime caveats from G0b apply until run on a batch-invariant GPU lane.

## 4. Constraints

- **Minimal diff is a gate, not a preference.** Report the total diff line
  count against upstream vLLM per phase. Target: Plane I ≈ config + ledger
  doc; Plane II ≈ plugins only; a fork of any core file is a design failure
  to be justified in writing or reverted.
- Pin the vLLM SHA; assert seams at import (`assert_seams()` pattern here).
- Preserve the two-function discipline. Any PR introducing a third hash or a
  non-canonical serialization into an identity path is wrong by definition.
- CPU lane = development truth; GPU lane (`VLLM_BATCH_INVARIANT=1`) = the
  lane for publishable throughput/byte-equality numbers.
- Store discipline is kappa-registry's: immutable κ blobs, tags as the only
  mutable tier, verify-on-read everywhere, negative-cache refused mappings.

## 5. Deliverables

1. `IDENTITY-LEDGER.md` — every identity, its preimage schema, its status
   (canonical / fixed / finding), and the external re-derivation proof.
2. Plugins (loader-verify, existing connector + drafter) with unit tests
   importable without torch (the lazy-class pattern in `connector.py`).
3. `bench/run_u*.py` harnesses, one per gate, each emitting one JSON line.
4. `MEASUREMENT-LOG.md` appended in this repo's dated, retraction-honest
   style — the log *is* the product; the code is its apparatus.
5. A closing verdict: one page stating, with numbers, which of
   {verification, latency, throughput, storage} UOR encoding measurably
   improves in vLLM, in which regime, and which claims must wait for the
   GPU lane. If an advantage is not demonstrated, the verdict says so.
