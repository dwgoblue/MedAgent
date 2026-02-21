from __future__ import annotations

from pathlib import Path

from medagent_system.runtime.harness.mvp_harness import run_mvp


def test_v2_harness_smoke(monkeypatch) -> None:
    monkeypatch.setenv("MEDAGENT_PIPELINE_MODE", "v2")
    monkeypatch.setenv("MEDAGENT_V2_ENABLE_CRITIC", "1")
    monkeypatch.setenv("MEDAGENT_V2_MAX_SUPERVISOR_REVISIONS", "1")

    cpb = Path("medagent_system/runtime/examples/sample_cpb.json")
    out = run_mvp(cpb)

    assert out["problem_list_ranked"]
    assert out["evidence_table"]
    assert out["provenance"]["pipeline"] == "medagent_system_v2_blackboard"
    assert "blackboard" in out["provenance"]
