from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

UTC = timezone.utc

from medagent.runtime.core.models import ClaimObject, DraftOutput, FinalOutput, SensitivityFinding


def synthesize_final_output(
    draft: DraftOutput,
    verified_claims: list[ClaimObject],
    sensitivity_map: list[SensitivityFinding],
    provenance: dict,
) -> FinalOutput:
    evidence_table = []
    unresolved_fail = False

    for claim in verified_claims:
        if claim.status == "fail" and claim.must_verify:
            unresolved_fail = True
        evidence_table.append(
            {
                "claim_text": claim.claim_text,
                "category": claim.category,
                "status": claim.status,
                "evidence_ids": [e.id for e in claim.evidence],
                "resolution": claim.resolution,
            }
        )

    final_soap = draft.soap_draft
    if any(c.status == "weak" and c.must_verify for c in verified_claims):
        final_soap += "\n\nCaveat: Some supporting claims are weak and retained with uncertainty language."

    guidance = "Escalate urgently if instability is high-risk or if evidence remains weak for critical claims."
    if unresolved_fail:
        guidance += " Unresolved failed claims detected; output should be blocked for production use."

    return FinalOutput(
        soap_final=final_soap,
        problem_list_ranked=draft.differential,
        plan_options_ranked_non_prescriptive=[
            "Collect additional focused diagnostics",
            "Prioritize evidence-supported differentials",
            "Reassess after new objective data",
        ],
        evidence_table=evidence_table,
        sensitivity_map=[asdict(s) for s in sensitivity_map],
        uncertainty_and_escalation_guidance=guidance,
        provenance={
            **provenance,
            "generated_at": datetime.now(UTC).isoformat(),
            "pipeline": "medagent_mvp_v1",
        },
    )
