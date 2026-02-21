from __future__ import annotations

import os
from typing import Any

from medagent_system.runtime.core.models import CPB
from medagent_system.runtime.core.models_v2 import BlackboardState, MedGemmaReport
from medagent_system.runtime.tools.medgemma import get_default_client, medgemma_enabled


def run_ehr_image_reporter_v2(cpb: CPB, state: BlackboardState, focus: list[str] | None = None) -> MedGemmaReport:
    focus = focus or []
    latest = cpb.timeline[-1]
    text = latest.ehr_text.strip()

    findings: list[str] = []
    assessments: list[str] = []
    obs: list[str] = []

    if text:
        findings.append(text[:500])
    image_refs: list[str] = []
    for img in latest.imaging[:5]:
        label = " | ".join(filter(None, [img.modality, img.body_part, (img.report_text or "").strip()]))
        if label:
            obs.append(label)
        image_refs.extend(img.refs[:3])

    used_model = False
    used_image_tensors = False
    if medgemma_enabled():
        try:
            client = get_default_client()
            system_prompt = (
                "You are an EHR+Image reporter. "
                "Return JSON only with keys: findings, candidate_assessments, supporting_observations. "
                "Never include genotype or literature claims."
            )
            prompt = (
                "Summarize this case as objective findings and hypotheses.\n\n"
                f"EHR text:\n{text}\n\n"
                f"Imaging observations:\n{chr(10).join(obs) if obs else 'No imaging descriptors'}\n\n"
                f"Revision focus: {', '.join(focus) if focus else 'none'}\n"
            )
            use_image_tensors = os.getenv("MEDAGENT_MEDGEMMA_USE_IMAGE_TENSORS", "0") == "1"
            if use_image_tensors and image_refs:
                data = client.generate_json_with_images(
                    prompt=prompt,
                    image_paths=image_refs,
                    system_prompt=system_prompt,
                    max_new_tokens=384,
                )
                used_image_tensors = data is not None
            else:
                data = client.generate_json(prompt, system_prompt=system_prompt, max_new_tokens=384)
            if data:
                findings = [str(x) for x in data.get("findings", []) if str(x).strip()][:6] or findings
                assessments = [str(x) for x in data.get("candidate_assessments", []) if str(x).strip()][:6]
                obs = [str(x) for x in data.get("supporting_observations", []) if str(x).strip()][:8] or obs
                used_model = True
        except Exception:
            used_model = False

    if not assessments:
        lower = text.lower()
        if "chest pain" in lower:
            assessments.append("Possible acute coronary syndrome (hypothesis)")
            obs.append("Chest pain documented in recent EHR narrative")
        if "shortness of breath" in lower or "dyspnea" in lower:
            assessments.append("Possible cardiopulmonary decompensation (hypothesis)")
            obs.append("Dyspnea/shortness of breath documented")
        if not assessments:
            assessments.append("Undifferentiated syndrome requiring additional objective data (hypothesis)")

    notes = []
    if focus:
        notes.append("Revision focus applied: " + "; ".join(focus[:3]))
    notes.append("No genotype assertions included by reporter policy")
    notes.append("No literature-based assertions included by reporter policy")
    if used_model:
        notes.append("Generated with local MedGemma model")
        if used_image_tensors:
            notes.append("Used image tensor inference from imaging refs")
    elif medgemma_enabled():
        notes.append("MedGemma enabled but inference fallback was applied")

    report = MedGemmaReport(
        findings=findings,
        candidate_assessments=assessments,
        supporting_observations=obs,
        notes=notes,
    )
    state.medgemma_report = report
    state.add_audit("a1_ehr_image_reporter", "write_medgemma_report", assessments=len(assessments))
    return report
