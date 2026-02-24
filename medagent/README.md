# medagent

Isolated implementation workspace for the actual MedGemma multi-agent runtime.

## Top-level layout

- `runtime/`: Biomni-style dynamic task runtime and typed tools.
- `factory/`: Superpowers-style engineering workflows, tests, and CI gates.
- `docs/`: architecture, evidence contracts, and threat/safety model.

## Alignment with existing repo

- SynthLab (`/synthlab`) remains source of synthetic longitudinal data and evaluation harness.
- Existing MCP settings are respected (`synthlab/configs/mcp.servers.json`).
- Runtime artifacts are designed to interoperate with existing schemas and MedGemma orchestration.

## Current execution mode

- MedGemma specialist is local/mock logic placeholder (intended to be replaced by local HF inference wiring).
- BioMCP verification supports:
  - Mock mode (default, offline-safe).
  - OpenAI reasoning + local RAG mode (set `MEDAGENT_USE_OPENAI=1` with `OPENAI_API_KEY`).

## Biomni-inspired architecture

- The runtime design is inspired by Biomni-style tool orchestration and execution loops.
- This repo does not include a vendored `Biomni/` directory.
- Optional adapter hook remains available at:
  - `medagent/runtime/tools/exec_env/biomni_adapter.py`
- Integration design notes:
  - `medagent/docs/biomni_integration_plan.md`

## Workflow reference

- Agentic pipeline (CPB, RAG, reasoning, tool-calling, synthesis):
  - `medagent/docs/agentic_workflow.md`
- V1 (MVP) + V2 (blackboard 6-agent) execution and migration plan:
  - `medagent/docs/v1_v2_execution_plan.md`

## Launch quickstart

Run from repo root:

```bash
export PYTHONPATH=.
export MEDAGENT_PIPELINE_MODE=mvp   # v1
# or
export MEDAGENT_PIPELINE_MODE=v2    # new blackboard pipeline
python3 medagent/runtime/run_mvp.py
```
