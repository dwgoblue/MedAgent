from __future__ import annotations

from medagent_system.runtime.core.models_v2 import BlackboardState, DraftSOAP


def run_supervisor_integrator_v2(state: BlackboardState, verifier_feedback: list[str] | None = None) -> DraftSOAP:
    verifier_feedback = verifier_feedback or []
    med = state.medgemma_report
    geno = state.genotype_report

    subjective = med.findings[0] if med and med.findings else "No subjective narrative available"

    objective_parts: list[str] = []
    if med and med.supporting_observations:
        objective_parts.extend(med.supporting_observations[:5])
    if state.patient_summary.key_labs:
        objective_parts.append(f"Labs available: {len(state.patient_summary.key_labs)}")
    if state.patient_summary.key_vitals:
        objective_parts.append(f"Vitals available: {len(state.patient_summary.key_vitals)}")
    objective = "; ".join(objective_parts) if objective_parts else "Objective evidence limited"

    differential = med.candidate_assessments[:5] if med else ["Undifferentiated syndrome"]

    assessment_lines = list(differential)
    if geno and geno.interpretations:
        assessment_lines.append("Genotype interpretation included as supporting evidence only")
    if verifier_feedback:
        assessment_lines.append("Verifier-requested updates addressed")
    assessment = "; ".join(assessment_lines)

    plan_items = [
        "Correlate symptoms with objective findings and timeline",
        "Prioritize evidence-backed differential items",
        "Collect missing objective data for unresolved uncertainty",
    ]
    if verifier_feedback:
        plan_items.append("Apply verifier patch list before release")
    if geno and geno.actionability_caveats:
        plan_items.append("Keep genotype-dependent recommendations caveated")

    draft = DraftSOAP(
        subjective=subjective,
        objective=objective,
        assessment=assessment,
        plan="; ".join(plan_items),
        differential_ranked=differential,
        open_questions=[
            "What key negatives are still missing?",
            "Any temporal mismatch between symptoms and imaging findings?",
            "Any high-risk recommendation lacking strong evidence?",
        ],
    )
    state.draft_soap = draft
    state.add_audit("a3_supervisor", "write_draft_soap", differential=len(draft.differential_ranked))
    return draft


def build_targeted_revision_requests(state: BlackboardState) -> tuple[list[str], list[str]]:
    if state.verifier_report.status == "PASS":
        return [], []

    a1_focus: list[str] = []
    a2_focus: list[str] = []
    for patch in state.verifier_report.patch_list:
        if patch.issue_type in {"cross_modal_conflict", "timeline_mismatch", "missing_negative"}:
            a1_focus.append(patch.message)
        if patch.issue_type in {"unsupported_claim", "high_risk_weak_evidence"}:
            a2_focus.append(patch.message)

    return a1_focus[:4], a2_focus[:4]
