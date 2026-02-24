from __future__ import annotations

from medagent.runtime.agents.causal_verifier.agent import run_counterfactual_sensitivity
from medagent.runtime.agents.critic.agent_v2 import has_high_stakes_claims, run_critic_v2
from medagent.runtime.agents.data_steward.agent import validate_and_normalize_cpb
from medagent.runtime.agents.evidence_assembler.agent_v2 import build_claims_ledger_v2
from medagent.runtime.agents.genotype_translator.interpreter_v2 import run_genotype_interpreter_v2
from medagent.runtime.agents.medgemma.reporter_v2 import run_ehr_image_reporter_v2
from medagent.runtime.agents.supervisor.alpha_v2 import (
    build_targeted_revision_requests,
    run_supervisor_integrator_v2,
)
from medagent.runtime.agents.synthesis.v2_adapter import build_final_output_from_blackboard
from medagent.runtime.agents.verifier.agent_v2 import run_verifier_v2
from medagent.runtime.core.config import RuntimeConfig
from medagent.runtime.core.models import CPB, DraftOutput, FinalOutput
from medagent.runtime.core.models_v2 import init_blackboard_from_cpb
from medagent.runtime.core.run_logger import RunLogger
from medagent.runtime.tools.medgemma import medgemma_enabled
from medagent.runtime.tools.retrieval.simple_rag import LocalRAGRetriever


class OrchestratorEngineV2:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig.from_env()
        self.retriever = LocalRAGRetriever(self.config.rag_roots)

    def run(self, cpb: CPB) -> FinalOutput:
        logger = RunLogger.from_env(mode="v2", patient_id=cpb.patient_id)
        cpb = validate_and_normalize_cpb(cpb)
        state = init_blackboard_from_cpb(cpb)
        if logger is not None:
            logger.log_agent_output(agent="orchestrator_v2", stage="init_blackboard", output=state.to_dict())

        a1_focus: list[str] = []
        a2_focus: list[str] = []
        feedback: list[str] = []

        if logger is not None:
            logger.log_comm(
                sender="orchestrator_v2",
                receiver="a1_ehr_image_reporter",
                kind="agent_prompt",
                prompt={"focus": a1_focus, "input_fields": ["ehr", "image"]},
            )
        run_ehr_image_reporter_v2(cpb, state, focus=a1_focus)
        if logger is not None:
            logger.log_agent_output(agent="a1_ehr_image_reporter", stage="medgemma_report", output=state.medgemma_report)

        if logger is not None:
            logger.log_comm(
                sender="orchestrator_v2",
                receiver="a2_genotype_interpreter",
                kind="agent_prompt",
                prompt={"focus": a2_focus, "input_fields": ["genotype", "patient_summary_keywords"]},
            )
        run_genotype_interpreter_v2(cpb, state, self.retriever, self.config, focus=a2_focus)
        if logger is not None:
            logger.log_agent_output(agent="a2_genotype_interpreter", stage="genotype_report", output=state.genotype_report)

        for revision in range(self.config.v2_max_supervisor_revisions + 1):
            state.revision_id = revision
            state.add_audit("orchestrator_v2", "revision_start", revision=revision)

            if logger is not None:
                logger.log_comm(
                    sender="orchestrator_v2",
                    receiver="a3_supervisor",
                    kind="agent_prompt",
                    prompt={
                        "revision": revision,
                        "verifier_feedback": feedback,
                        "inputs": ["medgemma_report", "genotype_report", "patient_summary"],
                    },
                )
            run_supervisor_integrator_v2(state, verifier_feedback=feedback)
            if logger is not None:
                logger.log_agent_output(agent="a3_supervisor", stage="draft_soap", output=state.draft_soap)

            if logger is not None:
                logger.log_comm(
                    sender="orchestrator_v2",
                    receiver="a4_evidence_assembler",
                    kind="agent_prompt",
                    prompt={"constraint": "no_new_claims", "input": "draft_soap"},
                )
            build_claims_ledger_v2(state)
            if logger is not None:
                logger.log_agent_output(agent="a4_evidence_assembler", stage="claims_ledger", output=state.claims_ledger)

            if logger is not None:
                logger.log_comm(
                    sender="orchestrator_v2",
                    receiver="a5_verifier",
                    kind="agent_prompt",
                    prompt={"checks": ["contradictions", "unsupported_claims", "timeline", "soap_required_fields"]},
                )
            verifier = run_verifier_v2(state, config=self.config)
            if logger is not None:
                logger.log_agent_output(agent="a5_verifier", stage="verifier_report", output=verifier)

            high_stakes_now = has_high_stakes_claims(state)
            if verifier.status == "PASS" and not high_stakes_now:
                state.add_audit("orchestrator_v2", "verification_pass", revision=revision)
                break

            if revision >= self.config.v2_max_supervisor_revisions:
                state.add_audit(
                    "orchestrator_v2",
                    "max_revisions_reached",
                    revision=revision,
                    high_stakes=high_stakes_now,
                    verifier_status=verifier.status,
                )
                break

            if verifier.status != "PASS":
                a1_focus, a2_focus = build_targeted_revision_requests(state)
                feedback = [p.message for p in verifier.patch_list[:6]]
            else:
                a1_focus = ["High-stakes narrative requires stronger objective grounding and explicit negatives."]
                a2_focus = ["High-stakes claim requires stronger genotype/literature support or confidence downgrade."]
                feedback = ["High-stakes claims require another upstream evidence-focused revision."]
                state.add_audit("orchestrator_v2", "high_stakes_revision_requested", revision=revision)

            if logger is not None:
                logger.log_comm(
                    sender="a5_verifier",
                    receiver="a3_supervisor",
                    kind="revision_feedback",
                    prompt=feedback,
                )
                logger.log_comm(
                    sender="a3_supervisor",
                    receiver="a1_ehr_image_reporter",
                    kind="targeted_revision_request",
                    prompt=a1_focus,
                )
                logger.log_comm(
                    sender="a3_supervisor",
                    receiver="a2_genotype_interpreter",
                    kind="targeted_revision_request",
                    prompt=a2_focus,
                )
            run_ehr_image_reporter_v2(cpb, state, focus=a1_focus)
            run_genotype_interpreter_v2(cpb, state, self.retriever, self.config, focus=a2_focus)

        high_stakes = has_high_stakes_claims(state)
        zero_confidence = bool(state.draft_soap is not None and state.draft_soap.confidence_score <= 0.0)
        need_critic = self.config.v2_enable_critic

        if need_critic:
            for _ in range(self.config.v2_max_critic_cycles):
                if logger is not None:
                    logger.log_comm(
                        sender="orchestrator_v2",
                        receiver="a6_critic",
                        kind="agent_prompt",
                        prompt={
                            "trigger": "enabled" if not (state.verifier_report.status != "PASS" or high_stakes or zero_confidence) else "persistent_fail_or_high_stakes_or_zero_confidence",
                            "inputs": ["draft_soap", "claims_ledger", "verifier_report"],
                        },
                    )
                run_critic_v2(state)
                if logger is not None:
                    logger.log_agent_output(agent="a6_critic", stage="critic_report", output=state.critic_report)

        if state.verifier_report.status != "PASS":
            state.escalate_to_doctor = True

        sensitivity_map: list[dict] = []
        if self.config.v2_enable_causal_postcheck and state.draft_soap is not None:
            draft = DraftOutput(
                soap_draft=(
                    f"S: {state.draft_soap.subjective}\n"
                    f"O: {state.draft_soap.objective}\n"
                    f"A: {state.draft_soap.assessment}\n"
                    f"P: {state.draft_soap.plan}"
                ),
                differential=state.draft_soap.differential_ranked,
                rationale_snippets=[c.reasoning for c in state.claims_ledger[:5]],
                uncertainty_flags=[c.uncertainty for c in state.claims_ledger if c.uncertainty][:5],
            )
            sensitivity = run_counterfactual_sensitivity(cpb, draft)
            sensitivity_map = [
                {
                    "problem": s.problem,
                    "sensitivity_class": s.sensitivity_class,
                    "fragile_on": s.fragile_on,
                }
                for s in sensitivity
            ]
            state.add_audit("orchestrator_v2", "causal_postcheck", findings=len(sensitivity_map))

        tooling = {
            "medgemma": "v2-reporter-medgemma-local" if medgemma_enabled() else "v2-reporter-heuristic",
            "biomcp": (
                "v2-policy-biomcp-sdk-strict-no-fallback"
                if self.config.use_biomcp_sdk
                else "v2-policy-local-rag"
            ),
            "verifier": "v2-openai-biomcp" if self.config.use_openai else "v2-rules-only",
            "critic_enabled": self.config.v2_enable_critic,
            "causal_postcheck": self.config.v2_enable_causal_postcheck,
        }
        final = build_final_output_from_blackboard(state, sensitivity_map=sensitivity_map, tooling=tooling)
        if logger is not None:
            logger.log_agent_output(agent="synthesis_v2_adapter", stage="final_output", output=final)
        return final
