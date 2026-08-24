# vllm-kappa

**The κ-connector: zero-fork vLLM plugins that make inference 100% κ-address
verifiable — and faster wherever work repeats.**

vLLM already derives a UOR κ-address for every KV block (its `sha256_cbor`
prefix-cache hash is byte-identical to `uor-addr`'s canonical CBOR address —
measured). This project completes that address space with values, witnesses, and a
network, using only public vLLM plugin seams and the
[hologram-fabric](https://github.com/Hologram-Technologies/hologram-fabric) HTTP
surface for all storage, addressing, and distribution:

| Layer | Mechanism | Gives |
|---|---|---|
| L0 | `VLLM_BATCH_INVARIANT=1` | canonical bytes — κ(request) → output becomes re-derivable fact |
| L1 | KV-connector plugin → κ-witness chain | every output byte challengeable at O(one block) cost |
| L2 | custom speculative proposer over a κ-store | verified-by-construction draft reuse; a malicious store can only slow you down, never corrupt a token |
| L3 | κ-keyed KV blocks shared via fabric | prefill paid once network-wide, spot-verified on arrival |

The core inversion: speculative decoding is the one protocol where verifying
untrusted compute makes you *faster* — so the compute-once κ network plugs into a
verifier vLLM already runs profitably.

**Status: Phase 0/1 unit lane green.** [PROMPT.md](PROMPT.md) is the complete
implementation prompt (pinned SHAs, seams, gates, kill criteria, traps);
[MEASUREMENT-LOG.md](MEASUREMENT-LOG.md) records gate results. Implemented so far:
`vllm_kappa/` (addressing mirror, two-tier κ-store with fail-open fabric client,
witness chain builder, `KappaConnector`, `KappaProposer`) and the standalone
auditor `verifier/kappa_verify.py` — G0a and G1a pass in unit form (900/900
single-byte tampers refused). Serving gates run on the Linux/WSL lane next.
Upstream vLLM is cloned and pinned, never forked — nothing here patches it.

```bash
python -m pip install -e .[test]
python -m pytest tests/ -q                    # unit lane
python -m verifier.kappa_verify selftest      # G1a tamper sweep
```

Serve with both plugins enabled (Linux, vLLM at the pinned SHA):

```bash
VLLM_BATCH_INVARIANT=1 VLLM_KAPPA_STORE_DIR=/var/kappa \
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-config '{"method":"custom","model":"vllm_kappa.drafter.KappaProposer","num_speculative_tokens":8}' \
  --kv-transfer-config '{"kv_connector":"KappaConnector","kv_role":"kv_both","kv_connector_module_path":"vllm_kappa.connector"}'
```

Audit any response afterwards, offline:

```bash
python -m verifier.kappa_verify verify --store /var/kappa
```

Related, same measurement discipline:
[uor-matmul benchmark fork](https://github.com/humuhumu33/uor-matmul) ·
[qvac-verified-inference](https://github.com/Hologram-Technologies/qvac-verified-inference)
