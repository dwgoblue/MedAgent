# Biomni Integration Plan for MedAgent

## Decision

Use the local `Biomni` clone as the execution substrate and tool-orchestration backbone, while keeping MedAgent-specific schemas, stop conditions, and safety policy in `medagent_system`.

## Mapping: MedAgent -> Biomni

- `OrchestratorPlannerAgent`: wraps Biomni `A1` planning/execution loop.
- `DataStewardAgent`: stays in MedAgent (schema/timeline QC around CPB).
- `GenotypeTranslatorAgent`: stays in MedAgent (domain-specific genotype constraints).
- `MedGemmaSpecialistAgent`: local HF MedGemma path (outside Biomni core).
- `BioMCPVerificationAgent`: Biomni MCP integration (`add_mcp`) + MedAgent claim contract.
- `CausalVerificationAgent`: MedAgent perturbation harness.
- `SynthesisTriageAgent`: MedAgent final assembly and stop-condition enforcement.

## Implementation strategy

1. Keep Biomni as external dependency via adapter (`runtime/tools/exec_env/biomni_adapter.py`).
2. Avoid changing Biomni source directly for now.
3. Register BioMCP servers through Biomni MCP config and keep MedAgent-level pass/weak/fail logic.
4. Keep `MEDAGENT_BIOMNI_LOAD_DATALAKE=0` by default for local dev speed.

## Environment variables

- `MEDAGENT_BIOMNI_REPO` (default `/home/daweilin/MedAgent/Biomni`)
- `MEDAGENT_BIOMNI_DATA` (default `/home/daweilin/MedAgent/.biomni_data`)
- `MEDAGENT_BIOMNI_LLM` (default `gpt-5.2`)
- `MEDAGENT_BIOMNI_SOURCE` (optional provider override)
- `MEDAGENT_BIOMNI_TOOL_RETRIEVER` (`1`/`0`)
- `MEDAGENT_BIOMNI_TIMEOUT_SECONDS` (default `600`)
- `MEDAGENT_BIOMNI_COMMERCIAL_MODE` (`1`/`0`)
- `MEDAGENT_BIOMNI_LOAD_DATALAKE` (`1` to allow default Biomni datalake download)

## Next coding milestones

1. Add an orchestrator mode that delegates planning/execution to `BiomniAdapter.go()`.
2. Add MCP config for BioMCP tools and bridge outputs into MedAgent `ClaimObject` format.
3. Wire local HF MedGemma inference into `MedGemmaSpecialistAgent` and keep Biomni for tool-heavy reasoning tasks.
