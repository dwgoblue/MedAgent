# MedAgent

User-facing onboarding for installing and running MedAgent (v1/v2), including MedGemma, SynthLab, and BioMCP workflows.
MedAgent is inspired by Biomni's agent/tool orchestration patterns, but this repository does not vendor the Biomni codebase.

For architecture and advanced implementation details, see:
- `medagent/README.md`
- `medagent/runtime/README.md`

## 1) Environment setup

Clone this repository along with its submodules:

```bash
git clone --recursive https://github.com/dwgoblue/MedAgent.git
cd MedAgent
```

<!-- If a submodule fails with **"Repository not found"** (e.g. private `synthlab`), the URL in `.gitmodules` may be wrong for your org. Set the correct URL and retry:

```bash
git config submodule.synthlab.url https://github.com/bschilder/synthlab.git 
git submodule update --init --recursive
``` -->

Create the unified `medagent` env using `conda` or `mamba`:

```bash
mamba env create -f environment.yml
mamba activate medagent
export PYTHONPATH=.
```

Optional dependency install refresh:

```bash
pip install -r requirements.txt
```

## 2) API key setup

### OpenAI (optional)

```bash
export OPENAI_API_KEY="<your_openai_key>"
```

### Hugging Face (for MedGemma download/access)

```bash
export HF_TOKEN="<your_hf_token>"
```

## 3) Quick tests

Run core test suite:

```bash
pytest -q medagent/factory/tests
```

Run v1-only regression tests:

```bash
pytest -q medagent/factory/tests/test_mvp_smoke.py \
         medagent/factory/tests/test_openai_fallback.py \
         medagent/factory/tests/test_synthlab_wrapper_unit.py
```

## 4) Local single-patient run

### v1 (MVP)

```bash
export MEDAGENT_PIPELINE_MODE=mvp
python3 medagent/runtime/run_mvp.py
```

### v2 (blackboard 6-agent)

```bash
export MEDAGENT_PIPELINE_MODE=v2
python3 medagent/runtime/run_mvp.py
```

## 5) BioMCP connectivity check

```bash
python3 medagent/runtime/check_biomcp_sdk.py \
  --intent GENERAL_LITERATURE_SUPPORT \
  --query "BRAF melanoma"
```

## 6) End-to-end agentic workflow (cluster or personal)

Use the same pipeline and flags either on a **personal machine** (e.g. 4× A100, no job scheduler) or on a **cluster with SLURM**.

### Personal cluster / bare metal (no job scheduler)

Run from the repo root. Same options as the SLURM script, plus optional `--cpus N`, `--gpus 0,1,2,3`, or `--ngpus N` (use first N GPUs). Default: 8 CPUs, all visible GPUs.

Script: `medagent/runtime/run_medagent_local.sh`

**v1**

```bash
./medagent/runtime/run_medagent_local.sh --mode v1
```

**v2 + MedGemma + BioMCP SDK + critic**

```bash
./medagent/runtime/run_medagent_local.sh \
  --mode v2 \
  --use-medgemma 1 \
  --hf-home $HOME/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it \
  --use-biomcp-sdk 1 \
  --enable-critic 1 \
  --max-supervisor-revisions 2 \
  --max-critic-cycles 1
```

**Use one GPU only** (e.g. leave others free):

```bash
./medagent/runtime/run_medagent_local.sh \
  --ngpus 1 \
  --mode v2 \
  --use-medgemma 1 \
  --hf-home $HOME/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it \
  --use-biomcp-sdk 1 \
  --enable-critic 1 \
  --max-supervisor-revisions 2 \
  --max-critic-cycles 1
```

**v2 multi-patient SynthLab**

```bash
./medagent/runtime/run_medagent_local.sh \
  --mode v2 \
  --use-medgemma 1 \
  --use-biomcp-sdk 1 \
  --synthlab-max-patients 5 \
  --synthlab-modalities fhir,genomics,notes,dicom \
  --synthlab-download-if-missing 1
```

**v2 benchmark run (5 patients)**

```bash
./medagent/runtime/run_medagent_local.sh \
  --mode v2 \
  --use-medgemma 1 \
  --use-medgemma-image-tensors 1 \
  --hf-home $HOME/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it \
  --use-openai 1 \
  --openai-model gpt-5.2 \
  --use-biomcp-sdk 1 \
  --enable-critic 1 \
  --max-supervisor-revisions 3 \
  --max-critic-cycles 1 \
  --benchmark-outcomes 1 \
  --benchmark-max-patients 5 \
  --benchmark-use-biomcp-gate 1 \
  --kg-backend dashboard \
  --synthlab-download-if-missing 1 \
  --synthlab-modalities fhir,genomics,notes,dicom
```

### Cluster with SLURM (optional)

If your cluster has SLURM, submit the same pipeline via:

- `medagent/runtime/run_medagent.sbat`

**v1**

```bash
sbatch medagent/runtime/run_medagent.sbat --mode v1
```

**v2 + MedGemma + BioMCP SDK + critic + revision loop=2**

```bash
sbatch medagent/runtime/run_medagent.sbat \
  --mode v2 \
  --use-medgemma 1 \
  --hf-home $HOME/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it \
  --use-biomcp-sdk 1 \
  --enable-critic 1 \
  --max-supervisor-revisions 2 \
  --max-critic-cycles 1
```

**v2 multi-patient SynthLab job**

```bash
sbatch medagent/runtime/run_medagent.sbat \
  --mode v2 \
  --use-medgemma 1 \
  --use-biomcp-sdk 1 \
  --synthlab-max-patients 5 \
  --synthlab-modalities fhir,genomics,notes,dicom \
  --synthlab-download-if-missing 1
```

**v2 benchmark run (5 patients, strict full pipeline)**

```bash
sbatch medagent/runtime/run_medagent.sbat \
  --mode v2 \
  --use-medgemma 1 \
  --use-medgemma-image-tensors 1 \
  --hf-home $HOME/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it \
  --use-openai 1 \
  --openai-model gpt-5.2 \
  --use-biomcp-sdk 1 \
  --enable-critic 1 \
  --max-supervisor-revisions 3 \
  --max-critic-cycles 1 \
  --benchmark-outcomes 1 \
  --benchmark-max-patients 5 \
  --benchmark-use-biomcp-gate 1 \
  --kg-backend dashboard \
  --synthlab-download-if-missing 1 \
  --synthlab-modalities fhir,genomics,notes,dicom
```

## 7) Outputs and logs

When using **SLURM**: job logs go to `log/medagent_<JOBID>.out` and `log/medagent_<JOBID>.err`. When using **run_medagent_local.sh**, stdout/stderr go to the terminal.

Run artifacts (same for both):
- `medagent/runtime/examples/cluster_runs/<run_id>/final_output.json`
- `medagent/runtime/examples/cluster_runs/<run_id>/logs/agent_outputs.jsonl`
- `medagent/runtime/examples/cluster_runs/<run_id>/logs/agent_comms.jsonl`
- `medagent/runtime/examples/cluster_runs/<run_id>/benchmark/benchmark_summary.json`
- `medagent/runtime/examples/cluster_runs/<run_id>/benchmark/benchmark_per_patient.jsonl`

## 8) Launch dashboard (server + tunnel)

Run on server:

```bash
conda activate medagent
streamlit run medagent/runtime/dashboard.py --server.port 8501 --server.address 127.0.0.1
```

From local machine:

```bash
ssh -N -L 8501:127.0.0.1:8501 $USER$@bamdev3
```

Open:
- `http://127.0.0.1:8501`

## 9) Optional terminal chat with MedGemma

```bash
./medagent/runtime/chat_medgemma.sh
```

## 10) Troubleshooting

- `FileNotFoundError: ~/.cache/synthlab/coherent`:
  use `--synthlab-download-if-missing 1` in `.sbat` or pre-download coherent dataset.
- BioMCP returns empty results:
  run `check_biomcp_sdk.py` first and inspect `debug` block.
- MedGemma fallback used:
  check `final_output.json` at `provenance.blackboard.medgemma_report.notes`.
