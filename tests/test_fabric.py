"""Two-tier store: verify-on-read, fail-open, off-path sync."""

import time

import pytest

from vllm_kappa.fabric import (
    FabricClient,
    IntegrityError,
    KappaStore,
    LocalStore,
)


def test_local_roundtrip(tmp_path):
    store = LocalStore(tmp_path)
    label = store.put(b"hello kappa")
    assert store.get(label) == b"hello kappa"
    assert label.startswith("blake3:")
    assert label in store.labels()


def test_verify_on_read_catches_corruption(tmp_path):
    store = LocalStore(tmp_path)
    label = store.put(b"pristine bytes")
    path = store._path(label)
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(IntegrityError):
        store.get(label)


def test_two_tier_counters(tmp_path):
    store = KappaStore(LocalStore(tmp_path))
    label = store.put(b"abc")
    assert store.get(label) == b"abc"
    assert store.get("blake3:" + "0" * 64) is None
    assert store.counters["hit_local"] == 1
    assert store.counters["miss"] == 1


def test_fabric_fail_open(tmp_path):
    # nothing listens on this port: every fabric op degrades, nothing raises
    fabric = FabricClient("http://127.0.0.1:9", timeout=0.2)
    store = KappaStore(LocalStore(tmp_path), fabric)
    label = store.put(b"survives fabric outage")
    assert store.get(label) == b"survives fabric outage"
    deadline = time.time() + 3
    while store._q.qsize() and time.time() < deadline:
        time.sleep(0.05)
    store.close()
    assert store.counters["fabric_sync_fail"] >= 1


def test_put_is_idempotent(tmp_path):
    store = LocalStore(tmp_path)
    assert store.put(b"x") == store.put(b"x")
