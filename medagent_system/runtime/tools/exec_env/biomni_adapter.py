from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BiomniAdapterConfig:
    repo_path: Path
    data_path: Path
    llm: str
    source: str | None = None
    use_tool_retriever: bool = True
    timeout_seconds: int = 600
    commercial_mode: bool = False
    expected_data_lake_files: list[str] | None = None

    @classmethod
    def from_env(cls) -> "BiomniAdapterConfig":
        repo_path = Path(os.getenv("MEDAGENT_BIOMNI_REPO", "/home/daweilin/medagent/Biomni"))
        data_path = Path(os.getenv("MEDAGENT_BIOMNI_DATA", "/home/daweilin/medagent/.biomni_data"))
        llm = os.getenv("MEDAGENT_BIOMNI_LLM", "gpt-5.2")
        source = os.getenv("MEDAGENT_BIOMNI_SOURCE") or None
        use_tool_retriever = os.getenv("MEDAGENT_BIOMNI_TOOL_RETRIEVER", "1") == "1"
        timeout_seconds = int(os.getenv("MEDAGENT_BIOMNI_TIMEOUT_SECONDS", "600"))
        commercial_mode = os.getenv("MEDAGENT_BIOMNI_COMMERCIAL_MODE", "0") == "1"

        # Empty list avoids default 11GB datalake download during dev.
        expected_data_lake_files: list[str] | None = []
        if os.getenv("MEDAGENT_BIOMNI_LOAD_DATALAKE", "0") == "1":
            expected_data_lake_files = None

        return cls(
            repo_path=repo_path,
            data_path=data_path,
            llm=llm,
            source=source,
            use_tool_retriever=use_tool_retriever,
            timeout_seconds=timeout_seconds,
            commercial_mode=commercial_mode,
            expected_data_lake_files=expected_data_lake_files,
        )


class BiomniAdapter:
    def __init__(self, config: BiomniAdapterConfig | None = None) -> None:
        self.config = config or BiomniAdapterConfig.from_env()
        self._agent = None
        self._ensure_importable()

    def _ensure_importable(self) -> None:
        repo_str = str(self.config.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def initialize(self) -> None:
        if self._agent is not None:
            return
        from biomni.agent import A1

        kwargs: dict[str, Any] = {
            "path": str(self.config.data_path),
            "llm": self.config.llm,
            "use_tool_retriever": self.config.use_tool_retriever,
            "timeout_seconds": self.config.timeout_seconds,
            "commercial_mode": self.config.commercial_mode,
            "expected_data_lake_files": self.config.expected_data_lake_files,
        }
        if self.config.source is not None:
            kwargs["source"] = self.config.source

        self._agent = A1(**kwargs)

    def add_mcp(self, config_path: str) -> None:
        self.initialize()
        self._agent.add_mcp(config_path=config_path)

    def go(self, prompt: str) -> str:
        self.initialize()
        result = self._agent.go(prompt)
        return str(result)

    def available(self) -> bool:
        try:
            self._ensure_importable()
            import biomni  # noqa: F401

            return True
        except Exception:
            return False
