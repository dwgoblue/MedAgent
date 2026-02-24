from __future__ import annotations

from pathlib import Path

from medagent.runtime.harness.mvp_harness import run_mvp


def test_mvp_harness_smoke() -> None:
    cpb = Path("medagent/runtime/examples/sample_cpb.json")
    out = run_mvp(cpb)

    assert out["problem_list_ranked"]
    assert out["evidence_table"]
    assert "soap_final" in out
    assert out["provenance"]["patient_id"] == "synthetic-0001"
