"""Plane II for model weights — κ-manifest, verify-at-load, refusal.

A model's identity in vLLM is a mutable path/name string (Identity Ledger
finding F4). This module gives it content: every tensor in every safetensors
file gets a BLAKE3 κ; the manifest is an identity-plane record
(sha256 over canonical CBOR — the uor-addr rule) binding the model root to
the full tensor κ list. `verify` recomputes every κ and returns a
certificate or the exact refusing tensor — Certified | Witness, no third
state.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import blake3
import cbor2

MANIFEST_NAME = "kappa-manifest.json"


def _tensor_kappas(st_path: Path) -> tuple[list[dict], int]:
    """Per-tensor BLAKE3 labels from a safetensors file (header-guided)."""
    with st_path.open("rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
        out = []
        for name, meta in sorted(header.items()):
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            h = blake3.blake3()
            remaining = end - begin
            while remaining:
                chunk = f.read(min(remaining, 1 << 24))
                h.update(chunk)
                remaining -= len(chunk)
            out.append(
                {
                    "tensor": name,
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                    "bytes": end - begin,
                    "kappa": "blake3:" + h.hexdigest(),
                }
            )
    return out, st_path.stat().st_size


def build_manifest(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    files = []
    total = 0
    for st in sorted(model_dir.glob("*.safetensors")):
        tensors, size = _tensor_kappas(st)
        total += size
        files.append({"file": st.name, "bytes": size, "tensors": tensors})
    body = {"v": 1, "t": "weights-manifest", "files": files}
    # identity-plane label of the manifest itself: the uor-addr rule
    ident = hashlib.sha256(cbor2.dumps(body, canonical=True)).hexdigest()
    return {**body, "identity": "sha256:" + ident, "total_bytes": total}


@dataclass
class Witness:
    file: str
    tensor: str
    expected: str
    actual: str


def verify(model_dir: str | Path, manifest: dict) -> tuple[bool, Witness | None, float]:
    """Recompute every tensor κ. Returns (certified, witness, seconds)."""
    t0 = time.perf_counter()
    model_dir = Path(model_dir)
    for frec in manifest["files"]:
        fresh, _ = _tensor_kappas(model_dir / frec["file"])
        want = {t["tensor"]: t["kappa"] for t in frec["tensors"]}
        for t in fresh:
            if want.get(t["tensor"]) != t["kappa"]:
                return False, Witness(
                    frec["file"],
                    t["tensor"],
                    want.get(t["tensor"], "<absent>"),
                    t["kappa"],
                ), time.perf_counter() - t0
    return True, None, time.perf_counter() - t0


def proven_boot_gate(model_dir: str | Path) -> float:
    """Refuse to proceed unless the stored manifest certifies. Returns
    verify seconds; raises on refusal — the engine must not boot on
    tampered weights."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / MANIFEST_NAME).read_text())
    ok, witness, secs = verify(model_dir, manifest)
    if not ok:
        raise RuntimeError(
            f"REFUSED: {witness.file}:{witness.tensor} bytes do not derive "
            f"their label ({witness.expected} != {witness.actual})"
        )
    return secs
