from __future__ import annotations

from medagent_system.runtime.core.models import CPB


def genotype_summary_and_query_bundle(cpb: CPB) -> tuple[str, list[dict[str, str]]]:
    variants = []
    for event in cpb.timeline:
        variants.extend(event.genomics.variants)

    if not variants:
        return "No genotype variants available.", []

    summary_lines = []
    bundle = []
    for v in variants[:10]:
        label = ", ".join(filter(None, [v.gene, v.hgvs, v.consequence]))
        if not label:
            label = "unspecified variant"
        summary_lines.append(f"- {label}")
        bundle.append(
            {
                "query_type": "variant",
                "gene": v.gene or "",
                "hgvs": v.hgvs or "",
            }
        )

    summary = "Genotype summary (supporting evidence only):\n" + "\n".join(summary_lines)
    return summary, bundle
