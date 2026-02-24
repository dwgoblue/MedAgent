from __future__ import annotations

from medagent.runtime.core.models import ClaimEvidence, ClaimObject, DraftOutput
from medagent.runtime.tools.openai_client.client import OpenAIReasoner
from medagent.runtime.tools.retrieval.simple_rag import LocalRAGRetriever


def extract_claims_from_draft(draft: DraftOutput) -> list[ClaimObject]:
    claims: list[ClaimObject] = []
    for dx in draft.differential:
        claims.append(
            ClaimObject(
                claim_text=f"Differential includes: {dx}",
                category="guideline",
                must_verify=True,
                evidence_requirements=["guideline_or_literature"],
                status="weak",
            )
        )

    for r in draft.rationale_snippets:
        if "Genotype summary" in r:
            claims.append(
                ClaimObject(
                    claim_text="Genotype signal contributes to prioritization",
                    category="variant",
                    must_verify=True,
                    evidence_requirements=["variant_annotation", "gene_disease_support"],
                    status="weak",
                )
            )
            break
    return claims


def verify_claims_with_mock_biomcp(claims: list[ClaimObject]) -> list[ClaimObject]:
    verified: list[ClaimObject] = []
    for c in claims:
        text = c.claim_text.lower()
        if "acute coronary syndrome" in text:
            c.status = "pass"
            c.evidence.append(
                ClaimEvidence(
                    source="mock_biomcp",
                    id="PMID:MOCK-ACS-2026",
                    title="Chest pain triage evidence summary",
                )
            )
            c.resolution = "Supported by mock evidence record"
        elif c.category == "variant":
            c.status = "weak"
            c.evidence.append(
                ClaimEvidence(
                    source="mock_biomcp",
                    id="CLINVAR:MOCK-VARIANT",
                    title="Variant annotation requires phenotype correlation",
                )
            )
            c.resolution = "Retained with caveat; not used as sole basis"
        else:
            c.status = "weak"
            c.resolution = "No strong evidence in mock retrieval"
        verified.append(c)
    return verified


def verify_claims_with_rag_reasoning(
    claims: list[ClaimObject],
    retriever: LocalRAGRetriever,
    rag_top_k: int = 3,
    openai_model: str = "gpt-5.2",
    return_meta: bool = False,
) -> list[ClaimObject] | tuple[list[ClaimObject], bool]:
    try:
        reasoner = OpenAIReasoner(model=openai_model)
    except Exception:
        fallback = verify_claims_with_mock_biomcp(claims)
        return (fallback, False) if return_meta else fallback

    verified: list[ClaimObject] = []
    for c in claims:
        snippets = retriever.retrieve(c.claim_text, top_k=rag_top_k)
        snippet_payload = [{"source": s.source, "text": s.text[:800]} for s in snippets]
        try:
            decision = reasoner.verify_claim(
                claim_text=c.claim_text,
                category=c.category,
                evidence_requirements=c.evidence_requirements,
                snippets=snippet_payload,
            )
        except Exception:
            fallback = verify_claims_with_mock_biomcp(claims)
            return (fallback, False) if return_meta else fallback
        c.status = decision.status
        c.resolution = decision.resolution

        for eid in decision.evidence_ids:
            c.evidence.append(ClaimEvidence(source="rag_reasoning", id=str(eid)))
        for s in snippets:
            c.evidence.append(ClaimEvidence(source="local_rag", id=s.source))

        # Ensure must-verify claims are never silently passed without any evidence handle.
        if c.must_verify and not c.evidence and c.status == "pass":
            c.status = "weak"
            c.resolution = "Downgraded: missing auditable evidence handles"
        verified.append(c)
    return (verified, True) if return_meta else verified
