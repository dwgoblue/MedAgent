from __future__ import annotations

from medagent_system.runtime.core.models import CPB


def summarize_imaging(cpb: CPB) -> str:
    findings = []
    for event in cpb.timeline:
        for img in event.imaging:
            bits = [img.modality or "unknown modality", img.body_part or "unknown body part"]
            if img.report_text:
                bits.append(img.report_text.strip())
            findings.append(" | ".join(bits))

    if not findings:
        return "No imaging findings available."
    return "Imaging summary:\n" + "\n".join(f"- {f}" for f in findings[:8])
