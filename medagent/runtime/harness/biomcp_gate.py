from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from medagent.runtime.tools.retrieval.biomcp_sdk_client import query_biomcp_sdk


@dataclass
class GateResult:
    grade: str
    result_count: int
    citations: list[dict[str, Any]] = field(default_factory=list)
    attempted_calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokset(text: str) -> set[str]:
    return {t for t in text.lower().split() if t}


def grade_prediction_with_biomcp(
    *,
    predicted_label: str,
    context: str,
    use_biomcp_sdk: bool = True,
    max_results: int = 3,
) -> GateResult:
    if not use_biomcp_sdk:
        return GateResult(grade="not_run", result_count=0)

    query = f"{predicted_label} clinical evidence relevance {context[:180]}"
    rows, debug = query_biomcp_sdk(intent="GENERAL_LITERATURE_SUPPORT", query=query, max_results=max_results)
    citations = [
        {
            "citation_id": r.citation_id,
            "source": r.source,
            "summary": r.summary[:220],
        }
        for r in rows
    ]

    if not rows:
        return GateResult(
            grade="fail",
            result_count=0,
            citations=[],
            attempted_calls=debug.attempted_calls,
            errors=debug.errors,
        )

    pred_tokens = _tokset(predicted_label)
    best_overlap = 0
    for row in rows:
        score = len(pred_tokens.intersection(_tokset(row.summary)))
        if score > best_overlap:
            best_overlap = score

    grade = "pass" if best_overlap >= 2 else "weak"
    return GateResult(
        grade=grade,
        result_count=len(rows),
        citations=citations,
        attempted_calls=debug.attempted_calls,
        errors=debug.errors,
    )
