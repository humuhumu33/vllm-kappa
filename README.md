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

**Status: specification.** [PROMPT.md](PROMPT.md) is the complete, self-contained
implementation prompt — pinned SHAs, the four vLLM seams, the fabric API contract,
phase gates, kill criteria, and the traps already paid for in prior measurement
campaigns. Upstream vLLM is cloned and pinned, never forked; `git diff` in the clone
stays empty by CI assertion.

Related, same measurement discipline:
[uor-matmul benchmark fork](https://github.com/humuhumu33/uor-matmul) ·
[qvac-verified-inference](https://github.com/Hologram-Technologies/qvac-verified-inference)
