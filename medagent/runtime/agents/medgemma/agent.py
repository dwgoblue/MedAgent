from __future__ import annotations

from typing import Any

from medagent.runtime.core.models import CPB, DraftOutput
from medagent.runtime.tools.medgemma import get_default_client, medgemma_enabled


def _heuristic_draft(cpb: CPB, genotype_summary: str, imaging_summary: str) -> DraftOutput:
    latest = cpb.timeline[-1]
    text = latest.ehr_text.lower()

    differential = []
    plan = []
    rationale = []
    uncertainty = []

    if "chest pain" in text:
        differential.append("Acute coronary syndrome")
        plan.append("Obtain serial ECG and troponin per protocol")
        rationale.append("Chest pain context requires urgent rule-out of cardiac ischemia")
    if "shortness of breath" in text or "dyspnea" in text:
        differential.append("Heart failure exacerbation")
        plan.append("Assess volume status and order chest imaging/labs")
        rationale.append("Dyspnea with cardiopulmonary symptoms requires decompensation assessment")

    if not differential:
        differential = ["Undifferentiated syndrome requiring further evaluation"]
        plan = ["Collect focused history, exam, and baseline diagnostics"]
        uncertainty.append("Limited signal in narrative text")

    soap = (
        "S: " + latest.ehr_text.strip() + "\n"
        "O: Structured data reviewed with multimodal context.\n"
        "A: " + "; ".join(differential) + "\n"
        "P: " + "; ".join(plan)
    )

    rationale.append(genotype_summary)
    rationale.append(imaging_summary)
    uncertainty.append("Genotype interpretation is supportive and must be evidence-verified")

    return DraftOutput(
        soap_draft=soap,
        differential=differential,
        rationale_snippets=rationale,
        uncertainty_flags=uncertainty,
    )


def _medgemma_draft(cpb: CPB, genotype_summary: str, imaging_summary: str) -> DraftOutput:
    latest = cpb.timeline[-1]
    client = get_default_client()
    system_prompt = (
        "You are MedGemma assisting with decision-support SOAP drafting. "
        "Return strict JSON with keys: soap_draft, differential, rationale_snippets, uncertainty_flags. "
        "Do not provide dosing or prescribing instructions."
    )
    prompt = (
        "Draft a concise SOAP note and differential from these inputs.\n\n"
        f"EHR text:\n{latest.ehr_text}\n\n"
        f"Imaging summary:\n{imaging_summary}\n\n"
        f"Genotype context (supporting only):\n{genotype_summary}\n\n"
        "Constraints:\n"
        "- Genotype is supportive evidence only.\n"
        "- Include uncertainty flags for missing information.\n"
        "- JSON only."
    )
    data: dict[str, Any] | None = client.generate_json(prompt, system_prompt=system_prompt, max_new_tokens=512)
    if not data:
        raw = client.generate_text(prompt, system_prompt=system_prompt, max_new_tokens=384)
        return DraftOutput(
            soap_draft=raw or f"S: {latest.ehr_text.strip()}",
            differential=["Undifferentiated syndrome requiring further evaluation"],
            rationale_snippets=[genotype_summary, imaging_summary],
            uncertainty_flags=["MedGemma returned non-JSON response; fallback parsing applied"],
        )

    soap = str(data.get("soap_draft", "")).strip() or f"S: {latest.ehr_text.strip()}"
    differential = data.get("differential", [])
    rationale = data.get("rationale_snippets", [])
    uncertainty = data.get("uncertainty_flags", [])
    if not isinstance(differential, list) or not differential:
        differential = ["Undifferentiated syndrome requiring further evaluation"]
    if not isinstance(rationale, list):
        rationale = [str(rationale)]
    if not isinstance(uncertainty, list):
        uncertainty = [str(uncertainty)]

    return DraftOutput(
        soap_draft=soap,
        differential=[str(x) for x in differential[:6]],
        rationale_snippets=[str(x) for x in rationale[:8]],
        uncertainty_flags=[str(x) for x in uncertainty[:8]],
    )


def generate_medgemma_draft(
    cpb: CPB,
    genotype_summary: str,
    imaging_summary: str,
) -> DraftOutput:
    if medgemma_enabled():
        try:
            return _medgemma_draft(cpb=cpb, genotype_summary=genotype_summary, imaging_summary=imaging_summary)
        except Exception as exc:
            fallback = _heuristic_draft(cpb=cpb, genotype_summary=genotype_summary, imaging_summary=imaging_summary)
            fallback.uncertainty_flags.append(f"MedGemma inference failed; fallback used: {type(exc).__name__}")
            return fallback
    return _heuristic_draft(cpb=cpb, genotype_summary=genotype_summary, imaging_summary=imaging_summary)
