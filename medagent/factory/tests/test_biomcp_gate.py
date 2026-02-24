from __future__ import annotations

from medagent.runtime.core.models_v2 import CitationRecord
from medagent.runtime.harness import biomcp_gate
from medagent.runtime.harness.biomcp_gate import grade_prediction_with_biomcp
from medagent.runtime.tools.retrieval.biomcp_sdk_client import BioMCPSDKDebug


def test_biomcp_gate_not_run() -> None:
    out = grade_prediction_with_biomcp(
        predicted_label="pulmonary embolism",
        context="dyspnea and chest pain",
        use_biomcp_sdk=False,
    )
    assert out.grade == "not_run"


def test_biomcp_gate_pass(monkeypatch) -> None:
    def fake_query(intent: str, query: str, max_results: int = 3):
        rows = [
            CitationRecord(
                citation_id="cit-1",
                intent=intent,
                query=query,
                source="pubmed",
                summary="Pulmonary embolism diagnosis and management evidence update",
                metadata={},
            )
        ]
        return rows, BioMCPSDKDebug(ok=True, attempted_calls=["x"], errors=[], result_count=1)

    monkeypatch.setattr(biomcp_gate, "query_biomcp_sdk", fake_query)
    out = grade_prediction_with_biomcp(
        predicted_label="pulmonary embolism",
        context="acute dyspnea",
        use_biomcp_sdk=True,
    )
    assert out.grade == "pass"
    assert out.result_count == 1


def test_biomcp_gate_fail(monkeypatch) -> None:
    def fake_query(intent: str, query: str, max_results: int = 3):
        return [], BioMCPSDKDebug(ok=False, attempted_calls=["x"], errors=["err"], result_count=0)

    monkeypatch.setattr(biomcp_gate, "query_biomcp_sdk", fake_query)
    out = grade_prediction_with_biomcp(
        predicted_label="stroke",
        context="focal deficits",
        use_biomcp_sdk=True,
    )
    assert out.grade == "fail"
