---
name: medagent-agentic-runtime-factory
description: Use this skill when designing or implementing the MedGemma + SynthLab multi-agent system with two-plane architecture (Research Runtime and Engineering Factory), including CPB/CCG/claim schemas, BioMCP evidence verification, counterfactual robustness checks, and final SOAP/problem-list/plan-option synthesis.
---

# MedAgent Runtime + Factory

## Use this skill when

- You are building or changing the MedGemma-based multi-agent runtime.
- You need strict separation between runtime execution and engineering/change workflows.
- You need evidence contracts, BioMCP verification, and causal robustness checks.

## Two-Plane Rule

- Plane A: Research Runtime solves tasks and generates artifacts.
- Plane B: Engineering Factory builds/changes tools, schemas, and tests.
- Runtime agents do not edit production code; all code changes go through Factory workflow.

## Canonical Artifacts

1) `CPB` (Canonical Patient Bundle)
- Longitudinal multimodal events: EHR text/structured, imaging refs, genomics, timestamps.

2) `CCG` (Clinical Concept Graph)
- Concept nodes with polarity, strength, modality source, time range, support/contradiction links.

3) `ClaimObject`
- Claim text/category, must-verify flag, evidence requirements, evidence records, pass/weak/fail status.

## Runtime Agent Contracts

1) `OrchestratorPlannerAgent`
- Creates plan, modality routing, evidence contract, budgets, stop criteria.

2) `DataStewardAgent`
- Validates/normalizes CPB and timeline consistency.

3) `GenotypeTranslatorAgent`
- Produces bounded MedGemma genotype summary + BioMCP query bundle.

4) `ImagingPreprocessorAgent`
- Normalizes imaging artifacts/reports for multimodal inference.

5) `MedGemmaSpecialistAgent`
- Produces draft SOAP, differential list, rationale snippets, uncertainty flags.

6) `BioMCPVerificationAgent`
- Extracts claims, retrieves evidence, marks pass/weak/fail, returns required edits/caveats.

7) `CausalVerificationAgent`
- Builds CCG, runs perturbations, computes stability, returns sensitivity map.

8) `SynthesisTriageAgent`
- Merges all outputs into final SOAP, ranked problems, ranked non-prescriptive plan options, evidence table, and escalation guidance.

## Runtime Stage Order

1. Input CPB (from user bundle or SynthLab generator).
2. Plan + evidence contract.
3. Parallel specialist work (steward/genotype/imaging/medgemma).
4. Causal verification (concept perturbations and stability metrics).
5. BioMCP verification (claim-level pass/weak/fail).
6. Final synthesis with caveats and unresolved uncertainty.
7. Stop only when contract conditions are met.

## Stop Conditions

- All `must_verify` claims are pass or explicitly caveated.
- No unresolved failed claim remains.
- Sensitivity-critical dependencies are verified or flagged uncertain.
- Final artifacts are emitted.

## File Layout

- Control-plane docs: `agentic/`
- Runtime implementation: `medagent/runtime/`
- Factory workflows/tests/ci: `medagent/factory/`
- Architecture docs: `medagent/docs/`

## Factory Requirements (for code changes)

- Design before coding.
- Use short executable plans with explicit file paths and verification steps.
- TDD cycle: RED -> GREEN -> REFACTOR.
- Run severity-gated review before merge.

## SynthLab Alignment Rules

- Treat `synthlab` as data/eval authority and keep integration contracts explicit.
- Keep offline-safe defaults; do not require network/data downloads in standard tests.
- Reuse MCP configuration conventions from `synthlab/configs/mcp.servers.json`.

## Read these references when needed

- Workflow details: `agentic/references/workflow.md`
- Existing SynthLab system contract: `synthlab/docs/AGENTIC_SYSTEM.md`
- MCP integration constraints: `synthlab/docs/MCP_INTEGRATION.md`
- BioMCP integration patterns: `synthlab/docs/BIOMCP_INTEGRATION.md`
