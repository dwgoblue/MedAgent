from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import importlib
import inspect
import re
from typing import Any

from medagent_system.runtime.core.models_v2 import CitationRecord

_GENE_RE = re.compile(r"\b[A-Z0-9]{2,10}\b")


@dataclass
class BioMCPSDKDebug:
    ok: bool
    attempted_calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    result_count: int = 0


def _extract_genes(query: str) -> list[str]:
    genes = [g for g in _GENE_RE.findall(query) if any(c.isalpha() for c in g)]
    # Heuristic filter for common non-gene tokens.
    blocked = {"CT", "MRI", "DNA", "RNA", "CAD", "SOAP"}
    return [g for g in genes if g not in blocked][:3]


def _extract_diseases(query: str) -> list[str]:
    q = query.lower()
    terms = [
        "melanoma",
        "hypertension",
        "coronary artery disease",
        "heart failure",
        "pulmonary embolism",
        "atelectasis",
        "cancer",
    ]
    return [t for t in terms if t in q][:3]


def _to_records(intent: str, query: str, payload: Any, max_results: int) -> list[CitationRecord]:
    rows: list[CitationRecord] = []

    def add_item(item: Any, idx: int) -> None:
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or item.get("id") or "BioMCP result")
            summary = str(item.get("abstract") or item.get("summary") or item.get("snippet") or title)
            source = str(item.get("source") or item.get("database") or "biomcp")
            rid = str(item.get("id") or item.get("pmid") or f"biomcp-{idx}")
        else:
            title = str(item)
            summary = title
            source = "biomcp"
            rid = f"biomcp-{idx}"

        rows.append(
            CitationRecord(
                citation_id=f"biomcp-{intent.lower()}-{idx}",
                intent=intent,
                query=query,
                source=source,
                summary=f"{title}: {summary}"[:700],
                metadata={"backend": "biomcp_sdk", "source_id": rid},
            )
        )

    items: list[Any]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            items = payload["results"]
        elif isinstance(payload.get("items"), list):
            items = payload["items"]
        else:
            items = [payload]
    else:
        items = [payload]

    for i, item in enumerate(items[:max_results], start=1):
        add_item(item, i)
    return rows


async def _query_biomcp_async(intent: str, query: str, max_results: int) -> list[CitationRecord]:
    from biomcp import BioMCPClient

    genes = _extract_genes(query)
    diseases = _extract_diseases(query)

    async with BioMCPClient() as client:
        # Intent-driven minimal routing.
        if intent == "TRIALS":
            try:
                payload = await client.trials.search(condition=(diseases[0] if diseases else query), max_results=max_results)
            except TypeError:
                payload = await client.trials.search(query=(diseases[0] if diseases else query))
            return _to_records(intent, query, payload, max_results)

        # Default to literature support.
        try:
            payload = await client.articles.search(
                genes=genes or None,
                diseases=diseases or None,
                max_results=max_results,
            )
        except TypeError:
            try:
                payload = await client.articles.search(query=query, max_results=max_results)
            except TypeError:
                payload = await client.articles.search(query=query)
        return _to_records(intent, query, payload, max_results)


async def _call_maybe_async(fn: Any, *args: Any, **kwargs: Any) -> Any:
    out = fn(*args, **kwargs)
    if inspect.isawaitable(out):
        return await out
    return out


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs
    accepted: dict[str, Any] = {}
    for name in sig.parameters:
        if name in kwargs:
            accepted[name] = kwargs[name]
    return accepted


def _resolve_dotted(path: str) -> Any | None:
    """
    Resolve dotted paths where some segments are modules and final segment is callable.
    Example: biomcp.articles.search.search_articles
    """
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        try:
            obj: Any = importlib.import_module(mod_name)
            rest = parts[i:]
            for seg in rest:
                obj = getattr(obj, seg)
            return obj
        except Exception:
            continue
    return None


def _collect_candidate_callables(paths: list[str]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for p in paths:
        obj = _resolve_dotted(p)
        if callable(obj):
            rows.append((p, obj))
    return rows


def _build_request_payload(query: str, genes: list[str], diseases: list[str], max_results: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "q": query,
        "genes": genes or None,
        "diseases": diseases or None,
        "max_results": max_results,
        "limit": max_results,
        "size": max_results,
    }
    return payload


def _try_make_request_object(module_path: str, payload: dict[str, Any]) -> Any:
    """
    Try to build a strongly-typed request object if biomcp module exposes one.
    Falls back to raw dict payload.
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        return payload

    # Prefer explicit common names first.
    candidate_names = [
        "PubmedRequest",
        "ArticleSearchRequest",
        "ArticlesSearchRequest",
        "SearchRequest",
        "TrialQuery",
    ]
    for name in candidate_names:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            sig = inspect.signature(cls)
            kwargs = {k: v for k, v in payload.items() if k in sig.parameters and v is not None}
            return cls(**kwargs)
        except Exception:
            continue

    # Fallback: any class ending in Request/Query.
    for name in dir(mod):
        if not (name.endswith("Request") or name.endswith("Query")):
            continue
        cls = getattr(mod, name, None)
        if cls is None or not inspect.isclass(cls):
            continue
        try:
            sig = inspect.signature(cls)
            kwargs = {k: v for k, v in payload.items() if k in sig.parameters and v is not None}
            return cls(**kwargs)
        except Exception:
            continue

    return payload


async def _query_biomcp_module_api_debug(
    intent: str,
    query: str,
    max_results: int,
) -> tuple[list[CitationRecord], BioMCPSDKDebug]:
    import biomcp

    genes = _extract_genes(query)
    diseases = _extract_diseases(query)
    dbg = BioMCPSDKDebug(ok=False)

    # Prefer trials endpoints only for trial intent.
    if intent == "TRIALS":
        trial_candidates = _collect_candidate_callables(
            [
                "biomcp.trials.search_trials",
                "biomcp.trials.search",
                "biomcp.trials.search.search_trials",
                "biomcp.trials.search.search",
            ]
        )

        for name, fn in trial_candidates:
            kwargs = {
                "query": diseases[0] if diseases else query,
                "condition": diseases[0] if diseases else query,
                "max_results": max_results,
            }
            try:
                call_kwargs = _filter_kwargs_for_callable(fn, kwargs)
                dbg.attempted_calls.append(f"{name}({','.join(call_kwargs.keys())})")
                payload = await _call_maybe_async(fn, **call_kwargs)
                rows = _to_records(intent, query, payload, max_results)
                dbg.ok = True
                dbg.result_count = len(rows)
                return rows, dbg
            except Exception as exc:
                dbg.errors.append(f"{name}: {type(exc).__name__}: {exc}")

        dbg.errors.append("No callable biomcp.trials.* search function succeeded")
        return [], dbg

    # Literature/general support path.
    article_candidates = _collect_candidate_callables(
        [
            "biomcp.articles.search_articles",
            "biomcp.articles.search",
            "biomcp.articles.search.search_articles",
            "biomcp.articles.search.search",
        ]
    )

    for name, fn in article_candidates:
        payload = _build_request_payload(query=query, genes=genes, diseases=diseases, max_results=max_results)
        kwargs = {
            "genes": genes or None,
            "diseases": diseases or None,
            "query": query,
            "q": query,
            "max_results": max_results,
            "request": _try_make_request_object("biomcp.articles.search", payload),
        }
        try:
            call_kwargs = _filter_kwargs_for_callable(fn, kwargs)
            dbg.attempted_calls.append(f"{name}({','.join(call_kwargs.keys())})")
            payload = await _call_maybe_async(fn, **call_kwargs)
            rows = _to_records(intent, query, payload, max_results)
            dbg.ok = True
            dbg.result_count = len(rows)
            return rows, dbg
        except Exception as exc:
            dbg.errors.append(f"{name}: {type(exc).__name__}: {exc}")

    dbg.errors.append("No callable biomcp.articles.* search function succeeded")
    return [], dbg


async def _query_biomcp_async_debug(intent: str, query: str, max_results: int) -> tuple[list[CitationRecord], BioMCPSDKDebug]:
    try:
        from biomcp import BioMCPClient
    except Exception as exc:
        # Installed biomcp package may expose module-level APIs instead.
        rows, dbg = await _query_biomcp_module_api_debug(intent=intent, query=query, max_results=max_results)
        dbg.errors.insert(0, f"BioMCPClient unavailable: {type(exc).__name__}: {exc}")
        return rows, dbg

    genes = _extract_genes(query)
    diseases = _extract_diseases(query)
    dbg = BioMCPSDKDebug(ok=False)

    async with BioMCPClient() as client:
        if intent == "TRIALS":
            try:
                dbg.attempted_calls.append("BioMCPClient.trials.search(condition,max_results)")
                payload = await client.trials.search(condition=(diseases[0] if diseases else query), max_results=max_results)
            except TypeError as exc:
                dbg.errors.append(f"trials.search signature mismatch: {type(exc).__name__}: {exc}")
                dbg.attempted_calls.append("BioMCPClient.trials.search(query)")
                payload = await client.trials.search(query=(diseases[0] if diseases else query))
            rows = _to_records(intent, query, payload, max_results)
            dbg.ok = True
            dbg.result_count = len(rows)
            return rows, dbg

        try:
            dbg.attempted_calls.append("BioMCPClient.articles.search(genes,diseases,max_results)")
            payload = await client.articles.search(
                genes=genes or None,
                diseases=diseases or None,
                max_results=max_results,
            )
        except TypeError as exc:
            dbg.errors.append(f"articles.search signature mismatch(genes/diseases): {type(exc).__name__}: {exc}")
            try:
                dbg.attempted_calls.append("BioMCPClient.articles.search(query,max_results)")
                payload = await client.articles.search(query=query, max_results=max_results)
            except TypeError as exc2:
                dbg.errors.append(f"articles.search signature mismatch(query,max_results): {type(exc2).__name__}: {exc2}")
                dbg.attempted_calls.append("BioMCPClient.articles.search(query)")
                payload = await client.articles.search(query=query)
        rows = _to_records(intent, query, payload, max_results)
        dbg.ok = True
        dbg.result_count = len(rows)
        return rows, dbg


def query_biomcp_sdk(intent: str, query: str, max_results: int = 3) -> tuple[list[CitationRecord], BioMCPSDKDebug]:
    try:
        rows, dbg = asyncio.run(_query_biomcp_async_debug(intent=intent, query=query, max_results=max_results))
        return rows, dbg
    except Exception as exc:
        dbg = BioMCPSDKDebug(ok=False, errors=[f"{type(exc).__name__}: {exc}"], result_count=0)
        return [], dbg


def biomcp_sdk_available() -> bool:
    try:
        import biomcp  # noqa: F401

        return True
    except Exception:
        return False
