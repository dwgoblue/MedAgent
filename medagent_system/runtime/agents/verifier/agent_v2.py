from __future__ import annotations

from medagent_system.runtime.core.config import RuntimeConfig
from medagent_system.runtime.core.models_v2 import BlackboardState, VerifierPatch, VerifierReport
from medagent_system.runtime.tools.openai_client.client import OpenAIReasoner
from medagent_system.runtime.tools.retrieval.biomcp_sdk_client import query_biomcp_sdk


def run_verifier_v2(state: BlackboardState, config: RuntimeConfig | None = None) -> VerifierReport:
    cfg = config or RuntimeConfig.from_env()
    patches: list[VerifierPatch] = []
    draft = state.draft_soap
    reasoner = OpenAIReasoner(model=cfg.openai_model) if cfg.use_openai else None

    if draft is None:
        patches.append(
            VerifierPatch(
                claim_id="N/A",
                issue_type="missing_soap_field",
                message="Draft SOAP missing",
                suggested_fix="Supervisor must produce SOAP before verification",
            )
        )
        report = VerifierReport(status="FAIL", patch_list=patches, check_summary={"missing_draft": True})
        state.verifier_report = report
        state.add_audit("a5_verifier", "fail_missing_draft")
        return report

    if getattr(draft, "confidence_score", 1.0) <= 0.0:
        patches.append(
            VerifierPatch(
                claim_id="N/A",
                issue_type="zero_confidence",
                message="Supervisor confidence score is 0.0 due to missing BioMCP evidence",
                suggested_fix="Route to critic review and clinician escalation path",
            )
        )

    # Required SOAP fields
    for field_name in ["subjective", "objective", "assessment", "plan"]:
        if not getattr(draft, field_name).strip():
            patches.append(
                VerifierPatch(
                    claim_id="N/A",
                    issue_type="missing_soap_field",
                    message=f"SOAP field missing: {field_name}",
                    suggested_fix=f"Populate SOAP {field_name} from available evidence",
                )
            )

    # Missing negatives
    if "no " not in draft.objective.lower() and "denies" not in draft.subjective.lower():
        patches.append(
            VerifierPatch(
                claim_id="N/A",
                issue_type="missing_negative",
                message="Objective/subjective narrative lacks explicit negatives",
                suggested_fix="Add key negatives or document that negatives are unavailable",
            )
        )

    # Timeline sanity
    timeline = state.patient_summary.timeline
    if len(timeline) >= 2:
        times = [t.get("t", "") for t in timeline]
        if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
            patches.append(
                VerifierPatch(
                    claim_id="N/A",
                    issue_type="timeline_mismatch",
                    message="Timeline appears unsorted",
                    suggested_fix="Reconcile event ordering and temporal references",
                )
            )

    # Claim support checks
    for claim in state.claims_ledger:
        if claim.claim_type in {"Inferred", "Recommended"} and not claim.evidence_items:
            patches.append(
                VerifierPatch(
                    claim_id=claim.claim_id,
                    issue_type="unsupported_claim",
                    message=f"Claim lacks evidence: {claim.claim_text[:120]}",
                    suggested_fix="Attach supporting evidence or downgrade confidence and uncertainty",
                )
            )

        if claim.claim_type == "Recommended" and claim.confidence == "Low":
            patches.append(
                VerifierPatch(
                    claim_id=claim.claim_id,
                    issue_type="high_risk_weak_evidence",
                    message="Recommendation retained with low confidence",
                    suggested_fix="Add guideline/literature support or keep as explicit hypothesis",
                )
            )

        if claim.claim_type in {"Inferred", "Recommended"} and cfg.use_biomcp_sdk:
            sdk_rows, _ = query_biomcp_sdk(
                intent="GENERAL_LITERATURE_SUPPORT",
                query=claim.claim_text,
                max_results=cfg.v2_biomcp_max_sources_per_claim,
            )
            if not sdk_rows:
                patches.append(
                    VerifierPatch(
                        claim_id=claim.claim_id,
                        issue_type="biomcp_empty",
                        message=f"BioMCP returned empty for claim: {claim.claim_text[:120]}",
                        suggested_fix="Set confidence to 0 and force critic review/escalation",
                    )
                )
                continue

            if reasoner is not None:
                snippets = [{"source": s.source, "text": s.summary[:800]} for s in sdk_rows]
                try:
                    decision = reasoner.verify_claim(
                        claim_text=claim.claim_text,
                        category=claim.claim_type.lower(),
                        evidence_requirements=["biomcp_sdk_support"],
                        snippets=snippets,
                    )
                except Exception as exc:
                    patches.append(
                        VerifierPatch(
                            claim_id=claim.claim_id,
                            issue_type="external_verifier_error",
                            message=f"OpenAI verifier error: {type(exc).__name__}",
                            suggested_fix="Retry verifier call or route to manual review",
                        )
                    )
                    continue
                if decision.status != "pass":
                    patches.append(
                        VerifierPatch(
                            claim_id=claim.claim_id,
                            issue_type="external_verification_" + decision.status,
                            message=f"External verification returned {decision.status}: {claim.claim_text[:120]}",
                            suggested_fix="Revise claim or keep as uncertainty with escalation",
                        )
                    )

    status = "PASS" if not patches else "FAIL"
    report = VerifierReport(
        status=status,
        patch_list=patches,
        check_summary={
            "claim_count": len(state.claims_ledger),
            "patch_count": len(patches),
        },
    )
    state.verifier_report = report
    state.add_audit("a5_verifier", "write_verifier_report", status=status, patches=len(patches))
    return report
