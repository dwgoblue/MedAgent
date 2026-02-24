from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class QueryMemoryRecord:
    intent: str
    query: str
    sdk_ok: bool
    result_count: int
    errors: list[str]


def _enabled() -> bool:
    return os.getenv("MEDAGENT_BIOMCP_QUERY_MEMORY", "1").strip() == "1"


def _memory_path() -> Path:
    p = os.getenv("MEDAGENT_BIOMCP_QUERY_MEMORY_PATH", ".medagent/biomcp_query_memory.jsonl")
    path = Path(p)
    return path if path.is_absolute() else Path.cwd() / path


def _tokens(s: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(s) if len(tok) > 2}


def _iter_records(limit: int = 5000) -> list[QueryMemoryRecord]:
    path = _memory_path()
    if not path.exists():
        return []

    rows: list[QueryMemoryRecord] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(rows) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                intent = str(obj.get("intent", "")).strip().upper()
                query = str(obj.get("query", "")).strip()
                if not intent or not query:
                    continue
                rows.append(
                    QueryMemoryRecord(
                        intent=intent,
                        query=query,
                        sdk_ok=bool(obj.get("sdk_ok", False)),
                        result_count=int(obj.get("result_count", 0) or 0),
                        errors=[str(e) for e in (obj.get("errors") or [])[:5]],
                    )
                )
    except Exception:
        return []
    return rows


def suggest_queries(*, intent: str, query: str, max_suggestions: int = 1) -> list[str]:
    if not _enabled() or max_suggestions <= 0:
        return []

    intent = intent.strip().upper()
    qtok = _tokens(query)
    if not qtok:
        return []

    scored: list[tuple[float, str]] = []
    for rec in _iter_records():
        if rec.intent != intent:
            continue
        rtok = _tokens(rec.query)
        overlap = len(qtok.intersection(rtok))
        if overlap <= 0:
            continue
        score = float(overlap)
        if rec.sdk_ok:
            score += 1.0
        score += min(rec.result_count, 5) * 0.2
        scored.append((score, rec.query))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[str] = []
    seen: set[str] = {query.strip().lower()}
    for _, candidate in scored:
        key = candidate.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= max_suggestions:
            break
    return out


def append_query_result(
    *,
    intent: str,
    query: str,
    sdk_ok: bool,
    result_count: int,
    errors: list[str] | None = None,
) -> None:
    if not _enabled():
        return
    intent = intent.strip().upper()
    query = query.strip()
    if not intent or not query:
        return

    path = _memory_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "intent": intent,
                        "query": query,
                        "sdk_ok": bool(sdk_ok),
                        "result_count": int(result_count),
                        "errors": [str(e) for e in (errors or [])[:5]],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    except Exception:
        return
