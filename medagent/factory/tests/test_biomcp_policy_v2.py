from __future__ import annotations

from medagent.runtime.core.models_v2 import CitationRecord
from medagent.runtime.tools.retrieval import biomcp_policy_v2
from medagent.runtime.tools.retrieval.biomcp_policy_v2 import RetrievalPolicy, retrieve_for_intent
from medagent.runtime.tools.retrieval.simple_rag import LocalRAGRetriever


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


def test_biomcp_query_memory_improves_sdk_query(monkeypatch, tmp_path) -> None:
    retriever = LocalRAGRetriever([])
    policy = RetrievalPolicy(max_retrieval_calls_per_claim=2, max_sources_per_claim=3)
    mem = tmp_path / "query_memory.jsonl"
    mem.write_text(
        '{"intent":"THERAPEUTIC_ACTIONABILITY","query":"braf melanoma targeted therapy guideline","sdk_ok":true,"result_count":3,"errors":[]}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDAGENT_USE_BIOMCP_SDK", "1")
    monkeypatch.setenv("MEDAGENT_V2_BIOMCP_NO_FALLBACK", "1")
    monkeypatch.setenv("MEDAGENT_BIOMCP_QUERY_MEMORY", "1")
    monkeypatch.setenv("MEDAGENT_BIOMCP_QUERY_MEMORY_PATH", str(mem))

    called: list[str] = []

    class _Dbg:
        def __init__(self, ok: bool, result_count: int) -> None:
            self.ok = ok
            self.result_count = result_count
            self.attempted_calls = ["x"]
            self.errors = []

    def _sdk(intent: str, query: str, max_results: int = 3):
        called.append(query)
        if "targeted therapy guideline" in query:
            return (
                [
                    CitationRecord(
                        citation_id="cit-memory-1",
                        intent=intent,
                        query=query,
                        source="biomcp",
                        summary="memory boosted result",
                    )
                ],
                _Dbg(True, 1),
            )
        return ([], _Dbg(False, 0))

    monkeypatch.setattr(biomcp_policy_v2, "query_biomcp_sdk", _sdk)

    citations = retrieve_for_intent(
        intent="THERAPEUTIC_ACTIONABILITY",
        query="braf melanoma actionability",
        retriever=retriever,
        existing_citations=[],
        policy=policy,
    )
    assert citations
    assert any("targeted therapy guideline" in q for q in called)
