from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class EvidenceSnippet:
    source: str
    text: str


class LocalRAGRetriever:
    def __init__(self, roots: list[Path], max_chars_per_file: int = 4000) -> None:
        self._max_chars = max_chars_per_file
        self._docs: list[EvidenceSnippet] = []
        self._load(roots)

    def _load(self, roots: list[Path]) -> None:
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")[: self._max_chars]
                except Exception:
                    continue
                if not text.strip():
                    continue
                self._docs.append(EvidenceSnippet(source=str(path), text=text))

    @staticmethod
    def _tokens(s: str) -> set[str]:
        return {tok.lower() for tok in _TOKEN_RE.findall(s) if len(tok) > 2}

    def retrieve(self, query: str, top_k: int = 3) -> list[EvidenceSnippet]:
        q = self._tokens(query)
        if not q:
            return []

        scored: list[tuple[int, EvidenceSnippet]] = []
        for doc in self._docs:
            d = self._tokens(doc.text)
            score = len(q.intersection(d))
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
