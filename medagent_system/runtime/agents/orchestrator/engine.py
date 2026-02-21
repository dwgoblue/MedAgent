from __future__ import annotations

from medagent_system.runtime.agents.biomcp_verifier.agent import (
    extract_claims_from_draft,
    verify_claims_with_rag_reasoning,
    verify_claims_with_mock_biomcp,
)
from medagent_system.runtime.agents.causal_verifier.agent import run_counterfactual_sensitivity
from medagent_system.runtime.agents.data_steward.agent import validate_and_normalize_cpb
from medagent_system.runtime.agents.genotype_translator.agent import genotype_summary_and_query_bundle
from medagent_system.runtime.agents.imaging_preproc.agent import summarize_imaging
from medagent_system.runtime.agents.medgemma.agent import generate_medgemma_draft
from medagent_system.runtime.agents.synthesis.agent import synthesize_final_output
from medagent_system.runtime.core.config import RuntimeConfig
from medagent_system.runtime.core.models import CPB, FinalOutput
from medagent_system.runtime.core.run_logger import RunLogger
from medagent_system.runtime.tools.retrieval.simple_rag import LocalRAGRetriever


class OrchestratorEngine:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig.from_env()
        self.retriever = LocalRAGRetriever(self.config.rag_roots)

    def run(self, cpb: CPB) -> FinalOutput:
        logger = RunLogger.from_env(mode="mvp", patient_id=cpb.patient_id)
        cpb = validate_and_normalize_cpb(cpb)
        if logger is not None:
            logger.log_agent_output(agent="data_steward", stage="normalized_cpb", output=cpb.to_dict())

        if logger is not None:
            logger.log_comm(
                sender="orchestrator_v1",
                receiver="genotype_translator",
                kind="agent_request",
                prompt={"task": "Summarize genotype and build query bundle"},
            )
        genotype_summary, query_bundle = genotype_summary_and_query_bundle(cpb)
        if logger is not None:
            logger.log_agent_output(
                agent="genotype_translator",
                stage="genotype_summary",
                output={"summary": genotype_summary, "query_bundle": query_bundle},
            )

        if logger is not None:
            logger.log_comm(
                sender="orchestrator_v1",
                receiver="imaging_preproc",
                kind="agent_request",
                prompt={"task": "Summarize imaging findings"},
            )
        imaging_summary = summarize_imaging(cpb)
        if logger is not None:
            logger.log_agent_output(agent="imaging_preproc", stage="imaging_summary", output=imaging_summary)

        if logger is not None:
            logger.log_comm(
                sender="orchestrator_v1",
                receiver="medgemma",
                kind="agent_prompt",
                prompt={
                    "task": "Generate SOAP draft",
                    "genotype_summary": genotype_summary,
                    "imaging_summary": imaging_summary,
                },
            )
        draft = generate_medgemma_draft(
            cpb=cpb,
            genotype_summary=genotype_summary,
            imaging_summary=imaging_summary,
        )
        if logger is not None:
            logger.log_agent_output(
                agent="medgemma",
                stage="draft_output",
                output={
                    "soap_draft": draft.soap_draft,
                    "differential": draft.differential,
                    "rationale_snippets": draft.rationale_snippets,
                    "uncertainty_flags": draft.uncertainty_flags,
                },
            )

        claims = extract_claims_from_draft(draft)
        if logger is not None:
            logger.log_agent_output(
                agent="biomcp_verifier",
                stage="extracted_claims",
                output=[c.claim_text for c in claims],
            )
        if self.config.use_openai:
            verified_claims, used_live = verify_claims_with_rag_reasoning(
                claims=claims,
                retriever=self.retriever,
                rag_top_k=self.config.rag_top_k,
                openai_model=self.config.openai_model,
                return_meta=True,
            )
            biomcp_mode = (
                f"openai_rag:{self.config.openai_model}"
                if used_live
                else "mocked-verifier-fallback"
            )
        else:
            verified_claims = verify_claims_with_mock_biomcp(claims)
            biomcp_mode = "mocked-verifier"
        if logger is not None:
            logger.log_agent_output(
                agent="biomcp_verifier",
                stage="verified_claims",
                output=[
                    {
                        "claim_text": c.claim_text,
                        "status": c.status,
                        "resolution": c.resolution,
                        "evidence_ids": [e.id for e in c.evidence],
                    }
                    for c in verified_claims
                ],
            )
        sensitivity_map = run_counterfactual_sensitivity(cpb, draft)
        if logger is not None:
            logger.log_agent_output(
                agent="causal_verifier",
                stage="sensitivity_map",
                output=[{"problem": s.problem, "sensitivity_class": s.sensitivity_class} for s in sensitivity_map],
            )

        must_verify = [c for c in verified_claims if c.must_verify]
        unresolved_fail = any(c.status == "fail" for c in must_verify)

        provenance = {
            "patient_id": cpb.patient_id,
            "query_bundle_size": len(query_bundle),
            "must_verify_count": len(must_verify),
            "unresolved_fail": unresolved_fail,
            "tooling": {
                "medgemma": "mocked-draft-agent",
                "biomcp": biomcp_mode,
                "causal_verifier": "heuristic-tier1",
            },
        }

        final = synthesize_final_output(
            draft=draft,
            verified_claims=verified_claims,
            sensitivity_map=sensitivity_map,
            provenance=provenance,
        )
        if logger is not None:
            logger.log_agent_output(agent="synthesis", stage="final_output", output=final)
        return final
