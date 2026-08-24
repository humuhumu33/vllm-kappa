"""Gates U3 (verified weights) + U4 (content dedup across a model family).

U3: build the κ-manifest for Qwen2.5-0.5B-Instruct, time verify (MB/s and
ratio vs engine boot), flip one byte in a scratch copy → the proven-boot
gate must refuse and name the exact tensor.

U4: per-tensor κ overlap between the Instruct model and its base
Qwen2.5-0.5B — does a full finetune share any content? Plus 1 MiB
fixed-chunk κ overlap as the coarse content-defined proxy. An honest 0% is
a publishable negative per PROMPT-UOR kill criteria.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import blake3

from vllm_kappa.weights import MANIFEST_NAME, build_manifest, proven_boot_gate


def hub_dir(repo: str) -> Path:
    root = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}"
    snaps = sorted((root / "snapshots").iterdir())
    return snaps[-1]


def chunk_kappas(model_dir: Path, size=1 << 20) -> set[str]:
    out = set()
    for st in sorted(model_dir.glob("*.safetensors")):
        with st.open("rb") as f:
            while True:
                b = f.read(size)
                if not b:
                    break
                out.add(blake3.blake3(b).hexdigest())
    return out


def main():
    inst = hub_dir("Qwen/Qwen2.5-0.5B-Instruct")
    base = hub_dir("Qwen/Qwen2.5-0.5B")

    # ---- U3 ----
    t0 = time.perf_counter()
    manifest = build_manifest(inst)
    build_s = time.perf_counter() - t0
    (inst / MANIFEST_NAME).write_text(json.dumps(manifest))
    verify_s = proven_boot_gate(inst)
    mb = manifest["total_bytes"] / 1e6

    scratch = Path.home() / "kappa-u3-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir()
    victim = sorted(inst.glob("*.safetensors"))[0]
    dst = scratch / victim.name
    shutil.copy(victim.resolve(), dst)
    data = bytearray(dst.read_bytes())
    data[len(data) // 2] ^= 0x01  # single bit, deep in some tensor
    dst.write_bytes(bytes(data))
    (scratch / MANIFEST_NAME).write_text(json.dumps(manifest))
    try:
        proven_boot_gate(scratch)
        refusal = None
    except RuntimeError as e:
        refusal = str(e)[:140]
    shutil.rmtree(scratch, ignore_errors=True)

    # ---- U4 ----
    m_base = build_manifest(base)
    tk_inst = {
        t["kappa"] for f in manifest["files"] for t in f["tensors"]
    }
    tk_base = {t["kappa"] for f in m_base["files"] for t in f["tensors"]}
    ck_inst = chunk_kappas(inst)
    ck_base = chunk_kappas(base)

    print(
        json.dumps(
            {
                "gate": "U3+U4",
                "u3_model_mb": round(mb, 1),
                "u3_manifest_build_s": round(build_s, 2),
                "u3_verify_s": round(verify_s, 2),
                "u3_verify_mb_s": round(mb / verify_s, 0),
                "u3_tamper_refused": refusal,
                "u4_tensors": [len(tk_base), len(tk_inst)],
                "u4_tensor_overlap": len(tk_inst & tk_base),
                "u4_chunks_1mib": [len(ck_base), len(ck_inst)],
                "u4_chunk_overlap": len(ck_inst & ck_base),
            }
        )
    )


if __name__ == "__main__":
    main()
