from __future__ import annotations

from medagent.runtime.tools.retrieval.biomcp_sdk_client import (
    biomcp_sdk_available,
    query_biomcp_sdk,
)


def test_biomcp_sdk_adapter_safe_without_sdk() -> None:
    rows, debug = query_biomcp_sdk("GENERAL_LITERATURE_SUPPORT", "BRAF melanoma", max_results=2)
    assert isinstance(rows, list)
    assert isinstance(debug.ok, bool)
    assert isinstance(debug.errors, list)


def test_biomcp_sdk_available_returns_bool() -> None:
    assert isinstance(biomcp_sdk_available(), bool)
