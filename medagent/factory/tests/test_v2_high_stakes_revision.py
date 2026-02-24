from __future__ import annotations

from medagent.runtime.agents.orchestrator.engine_v2 import OrchestratorEngineV2
from medagent.runtime.core.models import CPB, Genomics, TimelineEvent
from medagent.runtime.core.models_v2 import (
    ClaimLedgerEntry,
    DraftSOAP,
    MedGemmaReport,
    VerifierReport,
)


class _FakeLogger:
    def __init__(self) -> None:
        self.comms: list[dict] = []

    def log_agent_output(self, **kwargs) -> None:  # noqa: D401
        return None

    def log_comm(self, **kwargs) -> None:
        self.comms.append(kwargs)


def test_high_stakes_triggers_targeted_revision(monkeypatch) -> None:
    import medagent.runtime.agents.orchestrator.engine_v2 as mod

    fake_logger = _FakeLogger()
    monkeypatch.setattr(mod.RunLogger, "from_env", classmethod(lambda cls, mode, patient_id: fake_logger))
    monkeypatch.setattr(mod, "validate_and_normalize_cpb", lambda cpb: cpb)

    def fake_a1(cpb, state, focus=None):
        state.medgemma_report = MedGemmaReport(
            findings=["chest pain"],
            candidate_assessments=["acute coronary syndrome"],
            supporting_observations=["st depressions"],
            notes=[],
        )

    monkeypatch.setattr(mod, "run_ehr_image_reporter_v2", fake_a1)
    monkeypatch.setattr(mod, "run_genotype_interpreter_v2", lambda cpb, state, retriever, config, focus=None: None)

    monkeypatch.setattr(mod, "run_supervisor_integrator_v2", lambda state, verifier_feedback=None: state.__setattr__(
        "draft_soap",
        DraftSOAP(
            subjective="pain",
            objective="st depressions and no dyspnea",
            assessment="acute coronary syndrome",
            plan="observe",
            differential_ranked=["acute coronary syndrome"],
            open_questions=[],
        ),
    ))

    monkeypatch.setattr(
        mod,
        "build_claims_ledger_v2",
        lambda state: state.__setattr__(
            "claims_ledger",
            [
                ClaimLedgerEntry(
                    claim_id="CLM-1",
                    claim_text="Differential includes: acute coronary syndrome",
                    claim_type="Inferred",
                    confidence="Medium",
                    evidence_items=[],
                )
            ],
        ),
    )
    monkeypatch.setattr(
        mod,
        "run_verifier_v2",
        lambda state, config=None: state.__setattr__(
            "verifier_report", VerifierReport(status="PASS", patch_list=[], check_summary={})
        )
        or state.verifier_report,
    )

    call_count = {"n": 0}

    def fake_high_stakes(state) -> bool:
        call_count["n"] += 1
        return call_count["n"] == 1

    monkeypatch.setattr(mod, "has_high_stakes_claims", fake_high_stakes)
    monkeypatch.setattr(mod, "run_counterfactual_sensitivity", lambda cpb, draft: [])

    cpb = CPB(
        patient_id="p1",
        timeline=[
            TimelineEvent(
                t="2026-01-01T00:00:00+00:00",
                encounter_type="ed",
                ehr_text="chest pain",
                structured={},
                imaging=[],
                genomics=Genomics(vcf_ref=None, variants=[]),
            )
        ],
    )
    out = OrchestratorEngineV2().run(cpb)
    assert out.problem_list_ranked

    targeted = [c for c in fake_logger.comms if c.get("kind") == "targeted_revision_request"]
    receivers = {c.get("receiver") for c in targeted}
    assert "a1_ehr_image_reporter" in receivers
    assert "a2_genotype_interpreter" in receivers
