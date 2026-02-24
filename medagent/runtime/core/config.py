from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    use_openai: bool
    openai_model: str
    rag_top_k: int
    rag_roots: list[Path]
    pipeline_mode: str
    v2_max_supervisor_revisions: int
    v2_max_critic_cycles: int
    v2_enable_critic: bool
    v2_enable_causal_postcheck: bool
    use_biomcp_sdk: bool
    v2_biomcp_max_retrieval_calls_per_claim: int
    v2_biomcp_max_sources_per_claim: int
    v2_supervisor_evolve_mode: bool
    v2_supervisor_python_confidence: bool

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        use_openai = os.getenv("MEDAGENT_USE_OPENAI", "0").strip() == "1"
        openai_model = os.getenv("MEDAGENT_OPENAI_MODEL", "gpt-5.2")
        rag_top_k = int(os.getenv("MEDAGENT_RAG_TOP_K", "3"))

        default_roots = [
            Path("medagent/docs"),
            Path("synthlab/docs"),
        ]
        roots_env = os.getenv("MEDAGENT_RAG_ROOTS", "")
        if roots_env.strip():
            rag_roots = [Path(p.strip()) for p in roots_env.split(":") if p.strip()]
        else:
            rag_roots = default_roots

        pipeline_mode = os.getenv("MEDAGENT_PIPELINE_MODE", "mvp").strip().lower()
        if pipeline_mode not in {"mvp", "v2"}:
            pipeline_mode = "mvp"

        return cls(
            use_openai=use_openai,
            openai_model=openai_model,
            rag_top_k=rag_top_k,
            rag_roots=rag_roots,
            pipeline_mode=pipeline_mode,
            v2_max_supervisor_revisions=int(os.getenv("MEDAGENT_V2_MAX_SUPERVISOR_REVISIONS", "2")),
            v2_max_critic_cycles=int(os.getenv("MEDAGENT_V2_MAX_CRITIC_CYCLES", "1")),
            v2_enable_critic=os.getenv("MEDAGENT_V2_ENABLE_CRITIC", "0").strip() == "1",
            v2_enable_causal_postcheck=os.getenv("MEDAGENT_V2_ENABLE_CAUSAL_POSTCHECK", "1").strip() == "1",
            use_biomcp_sdk=os.getenv("MEDAGENT_USE_BIOMCP_SDK", "0").strip() == "1",
            v2_biomcp_max_retrieval_calls_per_claim=int(
                os.getenv("MEDAGENT_V2_BIOMCP_MAX_RETRIEVAL_CALLS_PER_CLAIM", "2")
            ),
            v2_biomcp_max_sources_per_claim=int(
                os.getenv("MEDAGENT_V2_BIOMCP_MAX_SOURCES_PER_CLAIM", "3")
            ),
            v2_supervisor_evolve_mode=os.getenv("MEDAGENT_V2_SUPERVISOR_EVOLVE_MODE", "0").strip() == "1",
            v2_supervisor_python_confidence=os.getenv("MEDAGENT_V2_SUPERVISOR_PYTHON_CONFIDENCE", "0").strip() == "1",
        )
