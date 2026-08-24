"""Two-tier κ-store: local content-addressed cache in front, hologram-fabric behind.

Rules (PROMPT.md §3-4): the hot path reads the local tier only; all fabric I/O
is off-path (a daemon thread drains a queue); every read re-derives the digest
(T1: integrity is never inferred from a name, always recomputed over bytes);
fabric being down degrades to vanilla behavior — never raises into serving.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path

import httpx

from .addressing import BYTES_CODEC, blake3_digest, kappa, parse_kappa

logger = logging.getLogger("vllm_kappa.fabric")


class IntegrityError(RuntimeError):
    """Bytes did not re-derive to their κ label."""


def _verify(label: str, data: bytes) -> bytes:
    codec, digest = parse_kappa(label)
    if codec != BYTES_CODEC or blake3_digest(data) != digest:
        raise IntegrityError(f"bytes do not derive {label}")
    return data


class LocalStore:
    """Content-addressed files: <root>/<hex[:2]>/<hex>. Verify on every read."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, label: str) -> Path:
        _, digest = parse_kappa(label)
        hexd = digest.hex()
        return self.root / hexd[:2] / hexd

    def put(self, data: bytes) -> str:
        label = kappa(blake3_digest(data), BYTES_CODEC)
        path = self._path(label)
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix(".tmp." + str(os.getpid()))
            tmp.write_bytes(data)
            tmp.replace(path)
        return label

    def get(self, label: str) -> bytes | None:
        path = self._path(label)
        if not path.exists():
            return None
        return _verify(label, path.read_bytes())

    def labels(self) -> list[str]:
        return [
            kappa(bytes.fromhex(p.name), BYTES_CODEC)
            for p in self.root.glob("??/*")
            if len(p.name) == 64
        ]


class KeyedIndex:
    """key-hex → κ-label mapping (files under <root>/index).

    Content addressing has no keyed lookup by design; this is the local
    naming tier. Its network counterpart is kappa-registry tags (/v2) —
    never grow a parallel protocol here beyond this file-per-key map.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root) / "index"
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: bytes, label: str) -> None:
        path = self.root / key.hex()
        tmp = path.with_suffix(".tmp." + str(os.getpid()))
        tmp.write_text(label)
        tmp.replace(path)

    def get(self, key: bytes) -> str | None:
        path = self.root / key.hex()
        try:
            return path.read_text().strip()
        except OSError:
            return None


class FabricClient:
    """Thin client for hologram-fabric's storage surface. Fail-open everywhere."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = 5.0,
    ):
        headers = {"X-Hologram-Token": token} if token else {}
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout
        )

    def put_object(self, data: bytes) -> str | None:
        try:
            r = self._client.post(
                "/api/v1/hologram/storage/objects",
                content=data,
                headers={"content-type": "application/octet-stream"},
            )
            r.raise_for_status()
            return r.json().get("address")
        except Exception as e:  # fail-open by contract
            logger.debug("fabric put failed (fail-open): %s", e)
            return None

    def get_object(self, label: str) -> bytes | None:
        try:
            r = self._client.get(f"/api/v1/hologram/storage/objects/{label}")
            r.raise_for_status()
            return _verify(label, r.content)
        except IntegrityError:
            raise  # a tampered store is a finding, not a network blip
        except Exception as e:
            logger.debug("fabric get failed (fail-open): %s", e)
            return None


class KappaStore:
    """local tier (hot, synchronous) + fabric tier (shared, off-path)."""

    def __init__(self, local: LocalStore, fabric: FabricClient | None = None):
        self.local = local
        self.fabric = fabric
        self.counters = {
            "put": 0,
            "hit_local": 0,
            "hit_fabric": 0,
            "miss": 0,
            "fabric_sync_fail": 0,
            "integrity_refused": 0,
        }
        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=4096)
        self._worker: threading.Thread | None = None
        if fabric is not None:
            self._worker = threading.Thread(
                target=self._drain, name="kappa-fabric-sync", daemon=True
            )
            self._worker.start()

    def _drain(self) -> None:
        while True:
            data = self._q.get()
            if data is None:
                return
            assert self.fabric is not None
            if self.fabric.put_object(data) is None:
                self.counters["fabric_sync_fail"] += 1

    def put(self, data: bytes) -> str:
        self.counters["put"] += 1
        label = self.local.put(data)
        if self.fabric is not None:
            try:
                self._q.put_nowait(data)
            except queue.Full:  # never block the hot path on sync backlog
                self.counters["fabric_sync_fail"] += 1
        return label

    def get(self, label: str) -> bytes | None:
        try:
            data = self.local.get(label)
        except IntegrityError:
            self.counters["integrity_refused"] += 1
            raise
        if data is not None:
            self.counters["hit_local"] += 1
            return data
        if self.fabric is not None:
            data = self.fabric.get_object(label)
            if data is not None:
                self.counters["hit_fabric"] += 1
                self.local.put(data)  # backfill the hot tier
                return data
        self.counters["miss"] += 1
        return None

    def close(self) -> None:
        if self._worker is not None:
            self._q.put(None)
            self._worker.join(timeout=2)


def store_from_env() -> KappaStore:
    """Build the store from VLLM_KAPPA_* environment (documented in README)."""
    root = os.getenv("VLLM_KAPPA_STORE_DIR", os.path.join(".", ".vllm-kappa"))
    fabric_url = os.getenv("VLLM_KAPPA_FABRIC_URL")
    fabric = None
    if fabric_url:
        fabric = FabricClient(fabric_url, token=os.getenv("VLLM_KAPPA_FABRIC_TOKEN"))
    return KappaStore(LocalStore(root), fabric)
