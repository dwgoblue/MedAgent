# Runtime

Biomni-style execution plane for patient-level multimodal analysis.

## Responsibilities

- Build plans and evidence contracts.
- Run specialist agents over CPB.
- Execute BioMCP claim verification.
- Execute causal perturbation robustness checks.
- Emit final artifacts after stop-condition checks.

## Agent directories

- `agents/orchestrator`
- `agents/data_steward`
- `agents/genotype_translator`
- `agents/imaging_preproc`
- `agents/medgemma`
- `agents/biomcp_verifier`
- `agents/causal_verifier`
- `agents/synthesis`

## MVP run

Run from repo root:

```bash
python3 medagent_system/runtime/run_mvp.py
```

Select pipeline mode:

```bash
export MEDAGENT_PIPELINE_MODE=mvp  # default
# or
export MEDAGENT_PIPELINE_MODE=v2   # blackboard 6-agent flow
```

## Cluster submit (.sbat)

Script:

- `medagent_system/runtime/run_medagent.sbat`
- Default resources in script are GPU-oriented (`gpuq`, `bio_ai`, `gpu:h100:1`).

Usage:

```bash
sbatch medagent_system/runtime/run_medagent.sbat v1
sbatch medagent_system/runtime/run_medagent.sbat v2
sbatch medagent_system/runtime/run_medagent.sbat v2 medagent_system/runtime/examples/sample_cpb.json medagent_system/runtime/examples/cluster_runs synthlab
```

Override resources at submit time if needed:

```bash
sbatch --gres=gpu:h100:2 --cpus-per-task=16 --mem=64G medagent_system/runtime/run_medagent.sbat --mode v2
```

Flag-based usage (no `export` needed):

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v2 \
  --env synthlab \
  --cpb medagent_system/runtime/examples/sample_cpb.json \
  --use-medgemma 1 \
  --hf-home /grid/koo/home/dalin/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it
```

Enable critic + set revision budget to 2:

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v2 \
  --enable-critic 1 \
  --max-supervisor-revisions 2 \
  --max-critic-cycles 1
```

Enable MedGemma image-tensor path (when imaging refs are present):

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v2 \
  --use-medgemma 1 \
  --use-medgemma-image-tensors 1 \
  --hf-home /grid/koo/home/dalin/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it
```

Run multi-patient SynthLab evaluation in one job:

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v2 \
  --synthlab-max-patients 5 \
  --synthlab-modalities fhir,genomics,notes,dicom
```

Run 1k outcome benchmark + BioMCP external gate:

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v2 \
  --benchmark-outcomes 1 \
  --benchmark-max-patients 1000 \
  --benchmark-use-biomcp-gate 1 \
  --synthlab-modalities fhir,genomics,notes,dicom
```

In benchmark mode, runtime defaults to strict full-run behavior:

- OpenAI verifier/supervisor enabled (`gpt-5.2` unless overridden)
- BioMCP SDK required with no local-RAG fallback (`MEDAGENT_V2_BIOMCP_NO_FALLBACK=1`)
- Critic enabled by default

OpenAI via flags:

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v1 \
  --use-openai 1 \
  --openai-model gpt-5.2
```

Enable BioMCP SDK retrieval in v2:

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v2 \
  --use-biomcp-sdk 1
```

Arguments:

1. mode: `v1|v2`
2. CPB path (optional)
3. output root directory (optional)
4. conda env name (optional, default `synthlab`)

## Run logging

Set `MEDAGENT_RUN_LOG_DIR` to enable per-run logging:

- `agent_outputs.jsonl`: structured outputs per agent stage
- `agent_comms.jsonl`: inter-agent communications/prompts
- `run_meta.json`: run metadata and file pointers

## Dashboard

View SOAP note, reports, and a knowledge graph for run outputs:

```bash
streamlit run medagent_system/runtime/dashboard.py
```

In the sidebar, provide a run JSON path such as:

- `medagent_system/runtime/examples/cluster_runs/<run_id>/synthlab_output.json`
- `medagent_system/runtime/examples/cluster_runs/<run_id>/final_output.json`

When present, the dashboard auto-loads:

- `logs/agent_outputs.jsonl`
- `logs/agent_comms.jsonl`

## BioMCP SDK smoke check

```bash
python3 medagent_system/runtime/check_biomcp_sdk.py --intent GENERAL_LITERATURE_SUPPORT --query "BRAF melanoma"
```

## SynthLab wrapper run

Run MedAgent over SynthLab multimodal patients and emit evaluation JSON:

```bash
python3 medagent_system/runtime/harness/synthlab_runner.py \
  --max-patients 1 \
  --modalities fhir,genomics,notes,dicom \
  --output medagent_system/runtime/examples/synthlab_run_output.json
```

If data is not cached and you want to allow download:

```bash
python3 medagent_system/runtime/harness/synthlab_runner.py \
  --max-patients 1 \
  --download-if-missing
```

## SynthLab 1k benchmark run

Run a large outcome benchmark directly (without sbatch wrapper):

```bash
python3 medagent_system/runtime/harness/synthlab_benchmark.py \
  --max-patients 1000 \
  --modalities fhir,genomics,notes,dicom \
  --use-biomcp-sdk 1 \
  --output-dir medagent_system/runtime/examples/benchmark_runs/latest \
  --download-if-missing
```

Artifacts:

- `benchmark_summary.json`
- `benchmark_per_patient.jsonl`

## OpenAI reasoning + local RAG (optional)

Enable API-backed claim verification:

```bash
export MEDAGENT_USE_OPENAI=1
export OPENAI_API_KEY=your_key
export MEDAGENT_OPENAI_MODEL=gpt-5.2
python3 medagent_system/runtime/run_mvp.py
```

Optional retrieval controls:

```bash
export MEDAGENT_RAG_TOP_K=3
export MEDAGENT_RAG_ROOTS=medagent_system/docs:synthlab/docs
```

## MedGemma local inference (optional)

Enable MedGemma in v1/v2 agents:

```bash
export MEDAGENT_USE_MEDGEMMA=1
export MEDAGENT_MEDGEMMA_MODEL_ID=google/medgemma-1.5-4b-it
export HF_HOME=/grid/koo/home/dalin/.cache/huggingface
export MEDAGENT_MEDGEMMA_LOCAL_ONLY=1
python3 medagent_system/runtime/run_mvp.py
```

If you prefer a direct model path:

```bash
export MEDAGENT_MEDGEMMA_MODEL_DIR=/grid/koo/home/dalin/.cache/huggingface/hub/models--google--medgemma-1.5-4b-it
```

Terminal chat launcher:

```bash
./medagent_system/runtime/chat_medgemma.sh
```

## Biomni adapter (optional)

Use local Biomni clone as orchestration/tool substrate:

```bash
export MEDAGENT_BIOMNI_REPO=/home/daweilin/MedAgent/Biomni
export MEDAGENT_BIOMNI_LLM=gpt-5.2
export MEDAGENT_BIOMNI_LOAD_DATALAKE=0
```

Adapter module:

- `medagent_system/runtime/tools/exec_env/biomni_adapter.py`
