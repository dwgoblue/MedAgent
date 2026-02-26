# MedAgent: A Reproducible Multi-agent System for Summarizing and Interpreting Multimodal Medical Data

![medagent](img/ChatGPT Image Feb 24, 2026, 06_17_18 PM.png)

**Authors**: Da Wei Lin & Brian M. Schilder (Cold Spring Harbor Laboratory)

## Introduction

MedAgent is a **reproducible multi-agent system** for clinical decision support. It orchestrates specialist agents—including [MedGemma](https://huggingface.co/google/medgemma) for imaging and reasoning, SynthLab for synthetic patient data, and BioMCP for literature and evidence—on a shared blackboard to produce evidence-backed assessments, [SOAP notes](https://www.ncbi.nlm.nih.gov/books/NBK482263/), and ranked care options. This repository was our submission to the [Kaggle Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups/medagent-a-reproducible-multi-agent-system). Making the pipeline open and reproducible matters for trust in AI-assisted medicine and for the community to build on MedGemma in real workflows.

# Setup

User-facing onboarding for installing and running MedAgent (v1/v2), including MedGemma, SynthLab, and BioMCP workflows, is below. MedAgent is inspired by Biomni's agent/tool orchestration patterns, but this repository does not vendor the Biomni codebase.

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

**v2 + MedGemma + BioMCP SDK + critic** (SynthLab pipeline; `--use-synthlab-soap` only applies when running SynthLab patients via `--synthlab-max-patients`)

```bash
./medagent/runtime/run_medagent_local.sh \
  --mode v2 \
  --use-medgemma 1 \
  --use-synthlab-soap 1 \
  --synthlab-max-patients 1 \
  --synthlab-download-if-missing 1 \
  --hf-home $HOME/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it \
  --use-biomcp-sdk 1 \
  --enable-critic 1 \
  --max-supervisor-revisions 2 \
  --max-critic-cycles 1
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

- **NumPy / faiss error** (`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` or `_ARRAY_API not found`):
  faiss-gpu is built for NumPy 1.x. Pin NumPy in the env: `pip install 'numpy>=1.26,<2'` (or recreate the env from `environment.yml`, which now pins `numpy<2`).
- `FileNotFoundError: ~/.cache/synthlab/coherent`:
  use `--synthlab-download-if-missing 1` in the runner or pre-download coherent dataset.
- **SynthLab S3 "No files found for fhir/genomics/..."**: S3 listing often returns empty from restricted networks (firewalls, VPNs, clusters). **Workaround**: download the full zip from a machine with S3 access, then unzip into the cache:
  `aws s3 cp --no-sign-request s3://synthea-open-data/coherent/coherent-11-07-2022.zip . && unzip coherent-11-07-2022.zip && mv coherent/unzipped/* ~/.cache/synthlab/coherent/`
  If data is already in `~/.cache/synthlab/coherent`, the run uses the cache and can still complete.
- BioMCP returns empty results:
  run `check_biomcp_sdk.py` first and inspect `debug` block.
- MedGemma fallback used:
  check `final_output.json` at `provenance.blackboard.medgemma_report.notes`.
- **SOAP notes short/vague vs original SynthLab**: By default, MedAgent builds SOAP from the supervisor (short summary of reporter + genotype). For **full, structured SOAP notes** matching the original SynthLab agentic pipeline (detailed S/O/A/P, patient story, assessment logic, future considerations), use SynthLab's SOAPNoteGenerator: pass `--use-synthlab-soap` to the runner, or set `MEDAGENT_USE_SYNTHLAB_SOAP=1`. Requires the SynthLab submodule and its MedGemma/SOAP dependencies; SOAP will be generated after the engine run and used as `soap_final`.
- **SynthLab SOAP: "or_mask_function / and_mask_function require torch>=2.6"**: The SynthLab SOAP generator (used with `--use-synthlab-soap`) goes through a code path that needs PyTorch ≥ 2.6. Upgrade with: `pip install 'torch>=2.6'` (or your CUDA-specific build, e.g. `pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cu124`).
- **SynthLab SOAP: "Could not import module 'AutoProcessor'. Are this object's requirements defined correctly?"**: Transformers is installed but AutoProcessor needs optional vision deps. Install with: `pip install 'transformers[vision]' pillow` (and ensure `torch` is installed). Run the script with the same Python (e.g. `conda activate medagent`).
- **SynthLab SOAP: "operator torchvision::nms does not exist"** (when importing AutoProcessor): Torch and torchvision are out of sync. Reinstall a matching pair: `pip install --upgrade torch torchvision`. If you use a CUDA build, install both from the same source (e.g. [PyTorch Get Started](https://pytorch.org/get-started/locally/): pick your CUDA version and run the given `pip install torch torchvision` command).
