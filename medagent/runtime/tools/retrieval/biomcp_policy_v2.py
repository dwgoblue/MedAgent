from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import os
from typing import Iterable

from medagent.runtime.core.models_v2 import CitationRecord
from medagent.runtime.core.run_logger import maybe_log_prompt_event
from medagent.runtime.tools.retrieval.biomcp_sdk_client import query_biomcp_sdk
from medagent.runtime.tools.retrieval.query_memory import append_query_result, suggest_queries
from medagent.runtime.tools.retrieval.simple_rag import LocalRAGRetriever

INTENTS = {
    "VARIANT_ANNOTATION",
    "GENE_DISEASE_ASSOCIATION",
    "PHENOTYPE_MATCH",
    "THERAPEUTIC_ACTIONABILITY",
    "TRIALS",
    "GENERAL_LITERATURE_SUPPORT",
}


@dataclass(frozen=True)
class RetrievalPolicy:
    max_retrieval_calls_per_claim: int = 2
    max_sources_per_claim: int = 3


def _norm_query(intent: str, query: str) -> str:
    return f"{intent.strip().upper()}::{query.strip().lower()}"


def _citation_id(intent: str, query: str, idx: int) -> str:
    digest = sha1(_norm_query(intent, query).encode("utf-8")).hexdigest()[:12]
    return f"cit-{digest}-{idx}"


def retrieve_for_intent(
    *,
    intent: str,
    query: str,
    retriever: LocalRAGRetriever,
    existing_citations: list[CitationRecord],
    policy: RetrievalPolicy,
) -> list[CitationRecord]:
    if intent not in INTENTS:
        intent = "GENERAL_LITERATURE_SUPPORT"

    normalized = _norm_query(intent, query)
    cached = [c for c in existing_citations if _norm_query(c.intent, c.query) == normalized]
    if cached:
        return cached[: policy.max_sources_per_claim]

    # Core ladder approximation in offline-safe mode:
    # 1) targeted retrieval, 2) broadened retrieval (synonyms/expansion).
    calls_budget = max(1, min(policy.max_retrieval_calls_per_claim, 2))
    queries: list[str] = [query]
    mem_queries = suggest_queries(intent=intent, query=query, max_suggestions=max(0, calls_budget - 1))
    maybe_log_prompt_event(
        sender="a2_genotype_interpreter",
        receiver="biomcp_query_memory",
        kind="planning",
        prompt={
            "intent": intent,
            "base_query": query,
            "calls_budget": calls_budget,
            "memory_hit": bool(mem_queries),
            "suggested_queries": mem_queries,
        },
    )
    for q in mem_queries:
        if q not in queries:
            queries.append(q)
    if len(queries) < calls_budget:
        expanded = query + " review guideline consensus"
        if expanded not in queries:
            queries.append(expanded)
    queries = queries[:calls_budget]

    use_biomcp_sdk = os.getenv("MEDAGENT_USE_BIOMCP_SDK", "0").strip() == "1"
    no_fallback = os.getenv("MEDAGENT_V2_BIOMCP_NO_FALLBACK", "1").strip() == "1"
    rows: list[CitationRecord] = []
    seen: set[str] = set()
    for step, q in enumerate(queries, start=1):
        snippets = []
        if use_biomcp_sdk:
            maybe_log_prompt_event(
                sender="a2_genotype_interpreter",
                receiver="biomcp_sdk",
                kind="tool_call",
                prompt={"intent": intent, "query": q, "max_results": policy.max_sources_per_claim},
            )
            sdk_rows, sdk_dbg = query_biomcp_sdk(intent=intent, query=q, max_results=policy.max_sources_per_claim)
            append_query_result(
                intent=intent,
                query=q,
                sdk_ok=bool(sdk_dbg.ok),
                result_count=int(sdk_dbg.result_count),
                errors=list(sdk_dbg.errors or []),
            )
            maybe_log_prompt_event(
                sender="biomcp_sdk",
                receiver="a2_genotype_interpreter",
                kind="tool_result",
                prompt={
                    "intent": intent,
                    "query": q,
                    "ok": sdk_dbg.ok,
                    "result_count": sdk_dbg.result_count,
                    "attempted_calls": sdk_dbg.attempted_calls,
                    "errors": sdk_dbg.errors,
                },
            )
            for rec in sdk_rows:
                key = f"{rec.source}:{rec.summary[:120]}"
                if key in seen:
                    continue
                seen.add(key)
                rec.metadata = {
                    **rec.metadata,
                    "ladder_step": step,
                    "sdk_ok": sdk_dbg.ok,
                    "sdk_attempted_calls": sdk_dbg.attempted_calls,
                    "sdk_errors": sdk_dbg.errors,
                }
                rows.append(rec)
                if len(rows) >= policy.max_sources_per_claim:
                    return rows

        if rows:
            continue

        if use_biomcp_sdk:
            maybe_log_prompt_event(
                sender="biomcp_sdk",
                receiver="a2_genotype_interpreter",
                kind="fallback_notice",
                prompt={"reason": "empty_or_failed_sdk_response", "query": q},
            )
            if no_fallback:
                continue
        snippets = retriever.retrieve(q, top_k=policy.max_sources_per_claim)
        for snip in snippets:
            key = f"{snip.source}:{snip.text[:120]}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                CitationRecord(
                    citation_id=_citation_id(intent, query, len(rows) + 1),
                    intent=intent,
                    query=query,
                    source=snip.source,
                    summary=snip.text[:500],
                    metadata={"ladder_step": step},
                )
            )
            if len(rows) >= policy.max_sources_per_claim:
                return rows
    return rows


def infer_intent_for_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["rs", "hgvs", "variant", "zygosity"]):
        return "VARIANT_ANNOTATION"
    if any(k in t for k in ["gene", "pathway", "association"]):
        return "GENE_DISEASE_ASSOCIATION"
    if any(k in t for k in ["phenotype", "symptom", "match"]):
        return "PHENOTYPE_MATCH"
    if any(k in t for k in ["trial", "clinical trial"]):
        return "TRIALS"
    if any(k in t for k in ["drug", "therapy", "actionable", "guideline"]):
        return "THERAPEUTIC_ACTIONABILITY"
    return "GENERAL_LITERATURE_SUPPORT"


def citations_to_evidence_summaries(citations: Iterable[CitationRecord]) -> list[str]:
    return [f"{c.source}: {c.summary[:160]}" for c in citations]
