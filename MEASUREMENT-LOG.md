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
