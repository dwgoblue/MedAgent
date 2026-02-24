from __future__ import annotations

from medagent.runtime.agents.biomcp_verifier.agent import (
    verify_claims_with_rag_reasoning,
    extract_claims_from_draft,
)
from medagent.runtime.core.models import DraftOutput
from medagent.runtime.tools.retrieval.simple_rag import LocalRAGRetriever


def test_openai_reasoning_falls_back_to_mock_without_client() -> None:
    draft = DraftOutput(
        soap_draft="S: chest pain\nO: data\nA: Acute coronary syndrome\nP: evaluate",
        differential=["Acute coronary syndrome"],
        rationale_snippets=["Genotype summary (supporting evidence only): AGT"],
        uncertainty_flags=[],
    )
    claims = extract_claims_from_draft(draft)
    retriever = LocalRAGRetriever([])

    verified = verify_claims_with_rag_reasoning(
        claims=claims,
        retriever=retriever,
        rag_top_k=2,
        openai_model="gpt-5.2",
    )

    assert verified
    assert {c.status for c in verified}.issubset({"pass", "weak", "fail"})
