from __future__ import annotations

from medagent_system.runtime.core.config import RuntimeConfig
from medagent_system.runtime.core.models import CPB
from medagent_system.runtime.core.models_v2 import (
    BlackboardState,
    GenotypeEvidence,
    GenotypeHypothesis,
    GenotypeReport,
)
from medagent_system.runtime.tools.retrieval.biomcp_policy_v2 import (
    RetrievalPolicy,
    citations_to_evidence_summaries,
    infer_intent_for_text,
    retrieve_for_intent,
)
from medagent_system.runtime.tools.retrieval.simple_rag import LocalRAGRetriever


def run_genotype_interpreter_v2(
    cpb: CPB,
    state: BlackboardState,
    retriever: LocalRAGRetriever,
    config: RuntimeConfig,
    focus: list[str] | None = None,
) -> GenotypeReport:
    focus = focus or []
    policy = RetrievalPolicy(
        max_retrieval_calls_per_claim=config.v2_biomcp_max_retrieval_calls_per_claim,
        max_sources_per_claim=config.v2_biomcp_max_sources_per_claim,
    )

    variants = []
    for event in cpb.timeline:
        variants.extend(event.genomics.variants)

    if not variants:
        report = GenotypeReport(
            interpretations=["No genotype variants available"],
            hypotheses=[],
            actionability_caveats=["No genotype-derived actionability conclusions"],
        )
        state.genotype_report = report
        state.add_audit("a2_genotype_interpreter", "write_genotype_report", variants=0)
        return report

    hypotheses: list[GenotypeHypothesis] = []
    interpretations: list[str] = []

    for v in variants[:8]:
        label = ", ".join(filter(None, [v.gene, v.hgvs, v.zygosity])) or "unspecified variant"
        query = f"{label} phenotype association clinical relevance"
        intent = infer_intent_for_text(query)
        citations = retrieve_for_intent(
            intent=intent,
            query=query,
            retriever=retriever,
            existing_citations=state.citations,
            policy=policy,
        )
        if citations:
            state.citations.extend([c for c in citations if c.citation_id not in {x.citation_id for x in state.citations}])

        summaries = citations_to_evidence_summaries(citations)
        evidence_items = [
            GenotypeEvidence(
                source_type="literature",
                source_id=c.citation_id,
                quote_or_summary=c.summary,
                method="qual",
                strength_level="moderate",
            )
            for c in citations
        ]

        phenotype_links = state.patient_summary.phenotype_keywords[:4]
        caveats = [
            "Genotype findings are supportive and must be integrated with phenotype and imaging",
            "Insufficient evidence remains possible and should be explicitly reported",
        ]
        if focus:
            caveats.append("Revision focus addressed: " + "; ".join(focus[:2]))

        confidence = "Low" if not citations else "Medium"
        interpretations.append(f"Variant hypothesis: {label} ({confidence} confidence)")
        if summaries:
            interpretations.append("Evidence summaries: " + " | ".join(summaries[:2]))
        hypotheses.append(
            GenotypeHypothesis(
                hypothesis=f"{label} may contribute to clinical phenotype",
                phenotype_links=phenotype_links,
                evidence_strength=confidence,
                caveats=caveats,
                evidence_items=evidence_items,
            )
        )

    report = GenotypeReport(
        interpretations=interpretations,
        hypotheses=hypotheses,
        actionability_caveats=[
            "No therapeutic action should be based on genotype alone",
            "Conflict across sources should downgrade confidence",
        ],
    )
    state.genotype_report = report
    state.add_audit(
        "a2_genotype_interpreter",
        "write_genotype_report",
        variants=len(variants[:8]),
        citations=len(state.citations),
    )
    return report
