from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medagent_system.runtime.agents.orchestrator.factory import build_orchestrator
from medagent_system.runtime.core.models import CPB


def load_cpb(path: Path) -> CPB:
    data: dict[str, Any] = json.loads(path.read_text())
    return CPB.from_dict(data)


def run_mvp(cpb_path: Path) -> dict[str, Any]:
    cpb = load_cpb(cpb_path)
    engine = build_orchestrator()
    result = engine.run(cpb)

    return {
        "soap_final": result.soap_final,
        "problem_list_ranked": result.problem_list_ranked,
        "plan_options_ranked_non_prescriptive": result.plan_options_ranked_non_prescriptive,
        "evidence_table": result.evidence_table,
        "sensitivity_map": result.sensitivity_map,
        "uncertainty_and_escalation_guidance": result.uncertainty_and_escalation_guidance,
        "provenance": result.provenance,
    }
