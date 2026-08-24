# IDENTITY-LEDGER — every identity vLLM mints, audited against the UOR rule

Gate U1 of PROMPT-UOR.md. The UOR rule for identities: `sha256(canonical-CBOR(preimage))`
(uor-addr's `cbor_address`, format `sha256:<64hex>`). For content bytes: `blake3(bytes)`
(kappa-registry). Audit of vLLM nightly `f94666b60` (the pinned SHA).

Status legend — **CANONICAL**: already the UOR rule (or becomes it with the
config flip). **PLANE-MIX**: correct but hashes bytes with sha256 where the
content plane says blake3. **FINDING**: not a content-derived identity; a
conformance gap with a proposed schema.

| # | Identity | Where minted | Preimage | Hash today | Status |
|---|---|---|---|---|---|
| 1 | **Block key** (prefix cache, chained) | `hash_block_tokens` | `(parent_key, block_token_ids, extra_keys)` | default `"sha256"` = sha256(**pickle**); opt-in `"sha256_cbor"` = the uor-addr rule | **CANONICAL after config flip.** G0a proved the `sha256_cbor` bytes ≡ uor-addr `cbor_address` ≡ this repo's torch-free mirror. Pickle default is NOT canonical (protocol-versioned). xxhash modes: non-crypto AND process-random root — never UOR. |
| 2 | **Root (NONE_HASH)** | `init_none_hash` | none — seeded | sha256 modes: deterministic; other modes: random per process | **CANONICAL under sha256 modes** (this repo pins `sha256(cbor("vllm-none-hash"))` as the known answer). Random-rooted modes mint identities that die with the process — F3. |
| 3 | **Multimodal item id** (`mm_hash`) | `multimodal/hasher.py` | raw mm content bytes | **blake3 by default** (sha256/sha512 for FIPS) | **CANONICAL** — vLLM independently chose the two-plane split this program uses: bytes → blake3. The strongest in-tree vindication of the design. |
| 4 | **mm extra key** in block hash | `_gen_mm_extra_hash_keys` | `(mm_identifier, offset_in_block)` | folded into #1 | **CANONICAL** with #1 — content-derived id + position, CBOR-encodable. |
| 5 | **LoRA extra key** | `_gen_lora_extra_hash_keys` | `lora_request.lora_name` — a **human-chosen string** | folded into #1 | **FINDING F1** — a mutable name, not content: two different adapters named alike collide the prefix cache; the same adapter renamed misses it. UOR schema: the adapter's weight-manifest identity (`weights.py` on the adapter dir) as the extra key. |
| 6 | **cache_salt** | request field, folded into block 0 | user-chosen string | folded into #1 | **CANONICAL by intent** — a deliberate identity input (cache partitioning), not a derivation. Documented, not a finding. |
| 7 | **Prompt-embeds key** | `_gen_prompt_embeds_extra_hash_keys` | raw embed tensor bytes | `hashlib.sha256(bytes)` | **PLANE-MIX F2** — bytes hashed with the identity-plane hash. Correct and stable, but the content plane says blake3 (as #3 does). Cosmetic; flag for upstream consistency. |
| 8 | **Model identity** | engine config | `model` **path/name string** | none — trust-by-path | **FINDING F4 — the big one.** The engine's own identity is a mutable label; nothing binds served results to weight content. Closed by Plane II: `weights.py` per-tensor blake3 κ + manifest identity `sha256:cbor(...)`; proven-boot gate refuses tampered bytes (U3: single-bit flip → named tensor refusal at 4.5 GB/s). |
| 9 | **Request seal / witness chain** | this repo (`witness.py`) | engine fingerprint + sampling fingerprint + token chain | sha256_cbor throughout | **CANONICAL** — built on the rule from day one; 16/16 real-request seals externally re-derived (G1c). |
| 10 | **KV payload label** | this repo (`connector.py`) | payload bytes | blake3 | **CANONICAL** (content plane); mapped to #1's key via the KeyedIndex — the bridge between input-addresses and byte-addresses. |
| 11 | **torch.compile cache key** | compilation config | config digest | md5/sha over internal repr | **OUT OF SCOPE** — a build-cache key, not an inference identity; never leaves the host. Listed for completeness. |

## Re-derivation proof (the U1 gate)

An identity is UOR only if a process with no vLLM import can re-derive it.
Status, on real artifacts from this repo's stores:

- **#1/#2/#4/#6**: `verifier/kappa_verify.py` + `tests/test_addressing.py`
  re-derive block keys from token streams using only the mirror rule —
  byte-equal to vLLM's output (G0a: known-answer vectors committed).
- **#9/#10**: the offline auditor re-derived 16/16 production seals and
  refuses tampered stores (G1a/G1c) and payloads (G3b/U7).
- **#3**: schema documented; re-derivation is `blake3(mm_bytes)` — not
  exercised on this lane (no mm model in the CPU harness). Open on GPU lane.
- **#5/#8**: not re-derivable **by design deficiency** — that is what makes
  them findings, closed by the schemas above (F1 proposal; F4 = `weights.py`, done).

**U1 verdict: 100% of identities minted on this lane are either re-derived
externally (with committed vectors) or converted into findings with concrete
UOR schemas — two of which (#8 via U3, #10 via G3) are already implemented
and measured in this repo.**
