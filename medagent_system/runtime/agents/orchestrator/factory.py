from __future__ import annotations

from medagent_system.runtime.agents.orchestrator.engine import OrchestratorEngine
from medagent_system.runtime.agents.orchestrator.engine_v2 import OrchestratorEngineV2
from medagent_system.runtime.core.config import RuntimeConfig


def build_orchestrator(config: RuntimeConfig | None = None) -> OrchestratorEngine | OrchestratorEngineV2:
    cfg = config or RuntimeConfig.from_env()
    if cfg.pipeline_mode == "v2":
        return OrchestratorEngineV2(cfg)
    return OrchestratorEngine(cfg)
