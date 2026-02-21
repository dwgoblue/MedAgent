from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from medagent_system.runtime.core.models import FinalOutput
from medagent_system.runtime.core.models_v2 import BlackboardState


def _soap_text(state: BlackboardState) -> str:
    draft = state.draft_soap
    if not draft:
        return ""
    return (
        f"S: {draft.subjective}\n"
        f"O: {draft.objective}\n"
        f"A: {draft.assessment}\n"
        f"P: {draft.plan}"
    )


def build_final_output_from_blackboard(
    state: BlackboardState,
    *,
    sensitivity_map: list[dict[str, Any]],
    tooling: dict[str, Any],
) -> FinalOutput:
    evidence_table: list[dict[str, Any]] = []
    failing_claims = {p.claim_id for p in state.verifier_report.patch_list}

    for claim in state.claims_ledger:
        status = "pass"
        if claim.claim_id in failing_claims:
            status = "fail" if claim.claim_type != "Observed" else "weak"
        elif claim.confidence == "Low":
            status = "weak"

        evidence_table.append(
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "category": claim.claim_type.lower(),
                "status": status,
                "evidence_ids": [e.source_id for e in claim.evidence_items],
                "resolution": claim.reasoning,
            }
        )

    guidance = "Escalate when evidence remains weak for high-impact recommendations."
    if state.escalate_to_doctor:
        guidance += " Escalation to medical doctors is required."

    return FinalOutput(
        soap_final=_soap_text(state),
        problem_list_ranked=state.draft_soap.differential_ranked if state.draft_soap else [],
        plan_options_ranked_non_prescriptive=[
            "Prioritize evidence-backed differential items",
            "Collect targeted objective diagnostics",
            "Reassess with reconciled cross-modal timeline",
        ],
        evidence_table=evidence_table,
        sensitivity_map=sensitivity_map,
        uncertainty_and_escalation_guidance=guidance,
        provenance={
            "patient_id": state.patient_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "pipeline": "medagent_system_v2_blackboard",
            "tooling": tooling,
            "blackboard": asdict(state),
        },
    )
