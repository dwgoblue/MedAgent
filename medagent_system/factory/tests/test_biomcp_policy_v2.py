from __future__ import annotations

from medagent_system.runtime.core.models_v2 import CitationRecord
from medagent_system.runtime.tools.retrieval import biomcp_policy_v2
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


def test_biomcp_policy_no_fallback_on_empty_sdk(monkeypatch) -> None:
    retriever = LocalRAGRetriever([])
    policy = RetrievalPolicy(max_retrieval_calls_per_claim=1, max_sources_per_claim=3)
    monkeypatch.setenv("MEDAGENT_USE_BIOMCP_SDK", "1")
    monkeypatch.setenv("MEDAGENT_V2_BIOMCP_NO_FALLBACK", "1")

    class _Dbg:
        ok = False
        result_count = 0
        attempted_calls = ["x"]
        errors = ["empty"]

    monkeypatch.setattr(biomcp_policy_v2, "query_biomcp_sdk", lambda intent, query, max_results=3: ([], _Dbg()))
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=3: [{"source": "local", "text": "fallback"}])

    citations = retrieve_for_intent(
        intent="GENERAL_LITERATURE_SUPPORT",
        query="acute coronary syndrome",
        retriever=retriever,
        existing_citations=[],
        policy=policy,
    )
    assert citations == []
