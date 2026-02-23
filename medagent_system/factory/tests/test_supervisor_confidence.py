from __future__ import annotations

from medagent_system.runtime.agents.supervisor.alpha_v2 import run_supervisor_integrator_v2
from medagent_system.runtime.core.models_v2 import BlackboardState, GenotypeReport, PatientSummary


def test_supervisor_sets_zero_confidence_on_biomcp_empty(monkeypatch) -> None:
    monkeypatch.setenv("MEDAGENT_USE_BIOMCP_SDK", "1")
    monkeypatch.setenv("MEDAGENT_USE_OPENAI", "0")
    state = BlackboardState(
        patient_id="p1",
        patient_summary=PatientSummary(),
        genotype_report=GenotypeReport(
            interpretations=["x"],
            hypotheses=[],
            actionability_caveats=[],
            biomcp_query_count=2,
            biomcp_empty_count=1,
        ),
    )
    draft = run_supervisor_integrator_v2(state)
    assert draft.confidence_score == 0.0
