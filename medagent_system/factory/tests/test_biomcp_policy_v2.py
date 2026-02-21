from __future__ import annotations

from medagent_system.runtime.core.models_v2 import CitationRecord
from medagent_system.runtime.tools.retrieval.biomcp_policy_v2 import RetrievalPolicy, retrieve_for_intent
from medagent_system.runtime.tools.retrieval.simple_rag import LocalRAGRetriever


def test_biomcp_policy_limits() -> None:
    retriever = LocalRAGRetriever([])
    policy = RetrievalPolicy(max_retrieval_calls_per_claim=2, max_sources_per_claim=3)

    citations = retrieve_for_intent(
        intent="GENERAL_LITERATURE_SUPPORT",
        query="acute coronary syndrome guideline",
        retriever=retriever,
        existing_citations=[],
        policy=policy,
    )

    assert len(citations) <= 3


def test_biomcp_policy_cache_hit() -> None:
    retriever = LocalRAGRetriever([])
    policy = RetrievalPolicy(max_retrieval_calls_per_claim=2, max_sources_per_claim=3)
    cached = [
        CitationRecord(
            citation_id="cit-1",
            intent="TRIALS",
            query="heart failure",
            source="cached",
            summary="cached summary",
        )
    ]

    citations = retrieve_for_intent(
        intent="TRIALS",
        query="heart failure",
        retriever=retriever,
        existing_citations=cached,
        policy=policy,
    )
    assert citations
    assert citations[0].source == "cached"
