from __future__ import annotations

from medagent_system.runtime.core.models import CPB, DraftOutput, SensitivityFinding


def run_counterfactual_sensitivity(cpb: CPB, draft: DraftOutput) -> list[SensitivityFinding]:
    findings: list[SensitivityFinding] = []

    for problem in draft.differential:
        if "Acute coronary syndrome" in problem:
            findings.append(
                SensitivityFinding(
                    problem=problem,
                    sensitivity_class="high",
                    fragile_on=["chest pain mention", "troponin abnormality"],
                )
            )
        else:
            findings.append(
                SensitivityFinding(
                    problem=problem,
                    sensitivity_class="medium",
                    fragile_on=["single-modality evidence"],
                )
            )

    return findings
