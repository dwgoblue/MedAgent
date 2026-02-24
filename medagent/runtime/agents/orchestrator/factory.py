from __future__ import annotations

from medagent.runtime.agents.orchestrator.engine import OrchestratorEngine
from medagent.runtime.agents.orchestrator.engine_v2 import OrchestratorEngineV2
from medagent.runtime.core.config import RuntimeConfig


def build_orchestrator(config: RuntimeConfig | None = None) -> OrchestratorEngine | OrchestratorEngineV2:
    cfg = config or RuntimeConfig.from_env()
    if cfg.pipeline_mode == "v2":
        return OrchestratorEngineV2(cfg)
    return OrchestratorEngine(cfg)
