from __future__ import annotations

from medagent_system.runtime.harness.synthlab_runner import _evaluate_run
from medagent_system.runtime.core.models import FinalOutput


def test_evaluate_run_basic() -> None:
    output = FinalOutput(
        soap_final="x",
        problem_list_ranked=["a"],
        plan_options_ranked_non_prescriptive=["p"],
        evidence_table=[
            {"category": "guideline", "status": "pass"},
            {"category": "variant", "status": "weak"},
        ],
        sensitivity_map=[{"x": 1}],
        uncertainty_and_escalation_guidance="g",
        provenance={"patient_id": "p1"},
    )
    eval_out = _evaluate_run(output)
    assert eval_out["must_verify_count"] == 2
    assert eval_out["must_verify_pass_rate"] == 0.5
    assert eval_out["must_verify_weak_rate"] == 0.5
