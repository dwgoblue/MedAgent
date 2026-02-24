from __future__ import annotations

from medagent.runtime.core.models_v2 import BlackboardState


HIGH_STAKES_TERMS = {
    "acute coronary syndrome",
    "stroke",
    "sepsis",
    "hemodynamic instability",
    "pulmonary embolism",
}


def has_high_stakes_claims(state: BlackboardState) -> bool:
    text = " ".join(c.claim_text.lower() for c in state.claims_ledger)
    return any(term in text for term in HIGH_STAKES_TERMS)


def run_critic_v2(state: BlackboardState) -> dict[str, object]:
    falsification_attempts = []
    evidence_demands = []

    for patch in state.verifier_report.patch_list[:8]:
        falsification_attempts.append(f"If {patch.claim_id} is removed, does assessment still hold?")
        evidence_demands.append(patch.suggested_fix)

    what_changes_mind = [
        "Objective negative findings that disconfirm leading diagnosis",
        "Higher-quality evidence supporting or refuting low-confidence recommendations",
        "Temporal reconciliation of symptom onset and modality findings",
    ]

    go_no_go = "GO" if state.verifier_report.status == "PASS" and not has_high_stakes_claims(state) else "NO_GO"
    report = {
        "falsification_attempts": falsification_attempts,
        "what_would_change_mind": what_changes_mind,
        "additional_evidence_demands": evidence_demands,
        "recommendation": go_no_go,
    }
    state.critic_report = report
    if go_no_go == "NO_GO":
        state.escalate_to_doctor = True
    state.add_audit("a6_critic", "write_critic_report", recommendation=go_no_go)
    return report
