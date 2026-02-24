# Agentic Workflow Reference

## Runtime stage order

1. Input CPB (from user or SynthLab generation)
2. Orchestrator plan + evidence contract + budgets
3. Parallel specialist work (data steward, genotype translator, imaging preprocessor, MedGemma specialist)
4. Causal verification (CCG extraction, perturbations, stability scoring)
5. BioMCP claim verification (pass/weak/fail)
6. Synthesis and triage with caveats and escalation guidance
7. Emit final artifacts + provenance
8. Enforce stop conditions

## Why synthesis is late-stage

SOAP/problem-list/plan options should summarize verified reasoning, not replace verification logic. Early synthesis increases risk of unsupported clinical claims.

## Minimal output contract

- Final SOAP note
- Ranked problem list
- Ranked non-prescriptive plan options
- Evidence table per claim/recommendation (or explicit caveat)
- Sensitivity map for critical concept dependencies
- Uncertainty and escalation guidance
- Provenance (`pipeline version`, `timestamp`, `model I/O trace`, tool versions)

## Stop conditions (required)

- All `must_verify` claims are pass or explicitly caveated.
- No unresolved failed claim remains.
- Sensitivity-critical dependencies are verified or flagged uncertain.
- Final artifact set is complete.

## Implementation locations

- Control-plane docs: `agentic/`
- Runtime implementation: `medagent/runtime/`
- Factory workflows/tests/ci: `medagent/factory/`
