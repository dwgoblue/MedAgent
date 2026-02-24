from __future__ import annotations

from medagent.runtime.core.models import FinalOutput
from medagent.runtime.harness import synthlab_runner


def _sample_output() -> FinalOutput:
    return FinalOutput(
        soap_final="S: chest pain\nO: objective\nA: ACS\nP: observe",
        problem_list_ranked=["acute coronary syndrome"],
        plan_options_ranked_non_prescriptive=["observe and reassess"],
        evidence_table=[{"claim_text": "st elevation suggests ACS", "category": "inferred", "status": "pass"}],
        sensitivity_map=[{"problem": "acute coronary syndrome", "sensitivity_class": "fragile"}],
        uncertainty_and_escalation_guidance="escalate when uncertain",
        provenance={},
    )


def test_dashboard_kg_backend() -> None:
    kg = synthlab_runner.build_kg_artifact(patient_id="p1", output=_sample_output(), kg_backend="dashboard")
    assert kg["backend"] == "dashboard"
    assert kg["nodes"]
    assert kg["edges"]


def test_synthlab_notebook_kg_backend(monkeypatch) -> None:
    class _Graph:
        def summary(self) -> str:
            return "CausalGraph: 1 nodes, 1 edges"

        def to_dict(self) -> dict:
            return {
                "nodes": [{"name": "p1", "type": "patient"}],
                "edges": [{"source": "p1", "target": "dx", "type": "++>"}],
            }

    class _SL:
        @staticmethod
        def parse_causal_graph(text: str):
            return _Graph()

    monkeypatch.setattr(synthlab_runner, "_import_synthlab", lambda: _SL())
    kg = synthlab_runner.build_kg_artifact(patient_id="p1", output=_sample_output(), kg_backend="synthlab_notebook")
    assert kg["backend"] == "synthlab_notebook"
    assert "graph" in kg
