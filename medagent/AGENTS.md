# AGENTS.md

Guidance for implementation work under `/home/daweilin/MedAgent/medagent`.

## Mission
Build the actual MedGemma-based multi-agent runtime with evidence-backed verification and counterfactual robustness, while staying aligned with SynthLab conventions.

## Plane Separation
- `runtime/`: task solving and artifact generation.
- `factory/`: code/schema changes, tests, and CI workflows.
- Runtime code must never self-modify production tools.

## Core Contracts
- Canonical artifacts: `CPB`, `CCG`, and `ClaimObject`.
- Verification-first output: every must-verify claim must be evidence-backed or caveated.
- Safety framing: decision-support only, no dosing/prescribing instructions.

## SynthLab Alignment
- Use SynthLab as the synthetic data and evaluation authority.
- Keep tests offline-safe by default.
- Reuse MCP config conventions in `synthlab/configs/mcp.servers.json`.

## Runtime Stop Conditions
- No unresolved failed must-verify claims.
- Sensitivity-critical dependencies verified or flagged uncertain.
- Final artifacts produced: SOAP, ranked problems, ranked non-prescriptive plan options, evidence table, sensitivity map, escalation guidance.
