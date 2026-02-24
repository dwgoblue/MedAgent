# SynthLab x MedGemma x BioMCP Agentic Architecture

## Two-plane model

- Research Runtime: dynamic orchestration for clinical reasoning tasks.
- Engineering Factory: disciplined code-change workflows, tests, and CI.

Runtime agents do not modify production code.

## Canonical artifacts

- `CPB`: canonical longitudinal multimodal patient bundle.
- `CCG`: time-aware concept graph for attribution and perturbation.
- `ClaimObject`: auditable verification units for BioMCP checks.

## Runtime flow

1. Input CPB from user/synth generator.
2. Orchestrator creates plan + evidence contract + budget.
3. Parallel specialist processing (steward/genotype/imaging/medgemma).
4. Causal verification perturbations and sensitivity scoring.
5. BioMCP claim verification pass/weak/fail.
6. Final synthesis with caveats and escalation guidance.
7. Stop when contract conditions are satisfied.

## Stop conditions

- All must-verify claims are pass or caveated.
- No unresolved failed claims.
- Sensitivity-critical dependencies verified or flagged uncertain.
- Final artifacts emitted.

## SynthLab alignment

- SynthLab remains source of synthetic data and core evaluation patterns.
- MCP integrations follow existing SynthLab config conventions.
- Offline-safe local tests remain default.
