# MedAgent V1 + V2 Execution Plan

Last updated: 2026-02-21

## Goal

Provide a stable **v1 workflow** and a parallel **v2 blackboard workflow** implementing the agreed 6-agent architecture for SynthLab + MedGemma + bioMCP with explicit claim traceability.

## Current status

- `v1` is the default (`MEDAGENT_PIPELINE_MODE=mvp`) and remains fully supported.
- `v2` is available behind mode switch (`MEDAGENT_PIPELINE_MODE=v2`) and keeps legacy `FinalOutput` compatibility.

## Pipelines

### V1 (MVP)

Flow:

1. Data steward normalizes CPB.
2. Genotype translator summarizes variants.
3. Imaging preprocessor summarizes imaging.
4. MedGemma placeholder drafts SOAP + differential.
5. BioMCP verifier extracts and scores claims.
6. Causal verifier runs perturbation sensitivity.
7. Synthesis emits final output.

Key traits:

- Linear pipeline.
- Claim status: `pass | weak | fail`.
- Output: `FinalOutput` with evidence table and guidance.

### V2 (Blackboard 6-agent)

Flow:

1. **A1 EHR+Image Reporter** writes `medgemma_report` (no genotype/literature claims).
2. **A2 Genotype Interpreter** writes `genotype_report` with intent-routed evidence retrieval.
3. **A3 Supervisor/Integrator** writes `draft_soap` from A1+A2+patient summary.
4. **A4 Evidence Assembler** builds `claims_ledger` (no new claims beyond draft).
5. **A5 Verifier** checks consistency/support/timeline and emits `verifier_report`.
6. If fail: bounded supervisor revisions; targeted requests to A1/A2.
7. Optional **A6 Critic** on persistent fail or high-stakes claims.
8. Optional causal post-check; add uncertainty/escalation only.

Key traits:

- Shared blackboard state with audit log and citation cache.
- Bounded loops (default max supervisor revisions = 2).
- Evidence policy limits per claim:
  - max retrieval calls: 2
  - max attached sources: 3

## Shared output contract

Both modes return legacy `FinalOutput` for harness compatibility:

- `soap_final`
- `problem_list_ranked`
- `plan_options_ranked_non_prescriptive`
- `evidence_table`
- `sensitivity_map`
- `uncertainty_and_escalation_guidance`
- `provenance`

In `v2`, full blackboard artifacts are embedded in `provenance.blackboard`.

## Configuration

Common:

- `MEDAGENT_PIPELINE_MODE` = `mvp` or `v2`
- `MEDAGENT_USE_OPENAI` = `0|1`
- `MEDAGENT_OPENAI_MODEL` (default `gpt-5.2`)
- `MEDAGENT_RAG_TOP_K` (default `3`)
- `MEDAGENT_RAG_ROOTS` (default `medagent/docs:synthlab/docs`)
- `MEDAGENT_USE_BIOMCP_SDK` (default `0`, set `1` to use biomcp-python in v2 retrieval)

V2-specific:

- `MEDAGENT_V2_MAX_SUPERVISOR_REVISIONS` (default `2`)
- `MEDAGENT_V2_MAX_CRITIC_CYCLES` (default `1`)
- `MEDAGENT_V2_ENABLE_CRITIC` (default `0`)
- `MEDAGENT_V2_ENABLE_CAUSAL_POSTCHECK` (default `1`)
- `MEDAGENT_V2_BIOMCP_MAX_RETRIEVAL_CALLS_PER_CLAIM` (default `2`)
- `MEDAGENT_V2_BIOMCP_MAX_SOURCES_PER_CLAIM` (default `3`)

## How to launch

From repo root:

```bash
export PYTHONPATH=.
```

Launch v1 single run:

```bash
export MEDAGENT_PIPELINE_MODE=mvp
python3 medagent/runtime/run_mvp.py
```

Launch v2 single run:

```bash
export MEDAGENT_PIPELINE_MODE=v2
python3 medagent/runtime/run_mvp.py
```

Run with explicit CPB/output path:

```bash
python3 medagent/runtime/run_mvp.py \
  --cpb medagent/runtime/examples/sample_cpb.json \
  --output medagent/runtime/examples/run_output.json
```

Launch SynthLab wrapper (either mode):

```bash
python3 medagent/runtime/harness/synthlab_runner.py \
  --max-patients 1 \
  --modalities fhir,genomics,notes,dicom \
  --output medagent/runtime/examples/synthlab_run_output.json
```

Cluster submit:

```bash
sbatch medagent/runtime/run_medagent.sbat v1
sbatch medagent/runtime/run_medagent.sbat v2
```

## How to test

V1 regression tests:

```bash
pytest -q medagent/factory/tests/test_mvp_smoke.py \
         medagent/factory/tests/test_openai_fallback.py \
         medagent/factory/tests/test_synthlab_wrapper_unit.py
```

V2 tests:

```bash
pytest -q medagent/factory/tests/test_v2_smoke.py \
         medagent/factory/tests/test_biomcp_policy_v2.py
```

Full factory test set:

```bash
pytest -q medagent/factory/tests
```

## Escalation policy

Escalate to doctor review when:

- verifier remains fail after max revisions,
- high-risk recommendations lack strong evidence,
- conflicting evidence affects high-impact decisions,
- required context is missing or modality is out-of-distribution.

## Logging outputs and communications

Set `MEDAGENT_RUN_LOG_DIR` to capture separate logs:

- `agent_outputs.jsonl` for per-agent outputs
- `agent_comms.jsonl` for inter-agent prompts/communications

Example:

```bash
export MEDAGENT_RUN_LOG_DIR=medagent/runtime/examples/logs/run1
export MEDAGENT_PIPELINE_MODE=v2
python3 medagent/runtime/run_mvp.py
```
