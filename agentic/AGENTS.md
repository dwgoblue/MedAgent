# AGENTS.md

Guidance for MedGemma-based multi-agent work under `/home/daweilin/MedAgent/agentic`.

## System Planes
- Plane A: Research Runtime (Biomni-style) executes tasks, tool orchestration, verification, and report generation.
- Plane B: Engineering Factory (Superpowers-style) owns code changes, schema evolution, tests, and CI gates.
- Rule: Runtime agents do not edit production tool code. Tool and schema changes go through Factory workflows.

## Scope Boundaries
- This folder defines control-plane architecture, contracts, and guardrails.
- Runtime implementation is isolated in `medagent/runtime`.
- Factory workflows are isolated in `medagent/factory`.
- SynthLab remains the authoritative synthetic data source and evaluation harness.

## Canonical Runtime Artifacts
1) Canonical Patient Bundle (`CPB`)
- Longitudinal multimodal patient payload: EHR, structured events, imaging refs, genotype refs, timestamps.

2) Clinical Concept Graph (`CCG`)
- Time-aware concept nodes with modality source, polarity, evidence strength, and support/contradiction links.

3) Claim Objects
- Auditable claim units with must-verify flags, evidence requirements, BioMCP evidence, and pass/weak/fail status.

## Runtime Agent Roster
1) `OrchestratorPlannerAgent`
- Builds plan, modality routing, evidence contract, budgets, and stop criteria.

2) `DataStewardAgent`
- Validates/normalizes CPB and enforces timeline/data quality checks.

3) `GenotypeTranslatorAgent`
- Produces bounded MedGemma genotype summary and structured BioMCP query bundle.

4) `ImagingPreprocessorAgent`
- Normalizes imaging/report artifacts for multimodal model input.

5) `MedGemmaSpecialistAgent`
- Produces draft SOAP, differential list, rationale snippets, and uncertainty flags.

6) `BioMCPVerificationAgent`
- Extracts claims, retrieves evidence, and marks claims pass/weak/fail with citations.

7) `CausalVerificationAgent`
- Runs concept perturbations and returns sensitivity map + critical dependency flips.

8) `SynthesisTriageAgent`
- Merges MedGemma + BioMCP + causal outputs into final artifacts.

## Stop Conditions (must all hold)
- Every `must_verify` claim is pass or explicitly caveated.
- No unresolved failed claim remains in final output.
- Sensitivity-critical dependencies are verified or clearly marked uncertain.
- Final artifacts are generated: SOAP, ranked problem list, ranked non-prescriptive plan options, evidence table, sensitivity map, escalation guidance.

## Safety and Output Policy
- Decision-support framing only; no prescribing or dosing instructions.
- Genomics is supporting evidence, never sole diagnostic basis.
- Always include provenance, uncertainty, and escalation guidance.

## SynthLab Alignment
- Preserve offline-safe defaults for local tests and CI.
- Reuse MCP configuration patterns from `synthlab/configs/mcp.servers.json`.
- Keep shared schema governance explicit and versioned.

## References
- `agentic/SKILL.md`
- `agentic/references/workflow.md`
- `synthlab/docs/AGENTIC_SYSTEM.md`
- `synthlab/docs/MCP_INTEGRATION.md`
- `synthlab/docs/BIOMCP_INTEGRATION.md`
