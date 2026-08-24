"""vllm-kappa: zero-fork κ-addressable verification and reuse for vLLM.

Nothing here modifies vLLM. Everything rides four public seams, asserted by
assert_seams() at plugin load so a vLLM version bump fails loudly instead of
silently corrupting witness identity.
"""

__version__ = "0.1.0"

PINNED_VLLM_SHA = "1baf372bf7c05a05ed4028f42d7d15b8d908a95a"


class SeamError(RuntimeError):
    """A vLLM public seam this plugin depends on has moved."""


def assert_seams() -> None:
    """Verify the four vLLM seams exist with the shapes we code against.

    Called by the connector and drafter constructors. Import-time cheap;
    raises SeamError with the pinned SHA so the operator knows what to diff.
    """
    problems: list[str] = []

    try:
        from vllm.utils.hashing import sha256_cbor  # noqa: F401
    except ImportError:
        problems.append("vllm.utils.hashing.sha256_cbor missing")

    try:
        import vllm.envs as envs

        if not hasattr(envs, "VLLM_BATCH_INVARIANT"):
            problems.append("vllm.envs.VLLM_BATCH_INVARIANT missing")
    except ImportError:
        problems.append("vllm.envs unimportable")

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        for method in (
            "start_load_kv",
            "save_kv_layer",
            "wait_for_save",
            "get_num_new_matched_tokens",
            "build_connector_meta",
        ):
            if not hasattr(KVConnectorBase_V1, method):
                problems.append(f"KVConnectorBase_V1.{method} missing")
    except ImportError:
        problems.append("kv_connector.v1.base unimportable")

    try:
        from vllm.v1.spec_decode.custom_class_proposer import (  # noqa: F401
            create_custom_proposer,
        )
    except ImportError:
        problems.append("custom_class_proposer.create_custom_proposer missing")

    if problems:
        raise SeamError(
            "vLLM seams moved (plugin pinned against "
            f"{PINNED_VLLM_SHA[:12]}): " + "; ".join(problems)
        )
