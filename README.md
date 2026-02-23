# MedAgent

User-facing onboarding for installing and running MedAgent (v1/v2), including MedGemma, SynthLab, and BioMCP workflows.

For architecture and advanced implementation details, see:
- `medagent_system/README.md`
- `medagent_system/runtime/README.md`

## 1) Environment setup

Clone this repository along with its submodules: 

```bash
git clone --recursive https://github.com/dwgoblue/MedAgent.git
cd MedAgent
```

Use your existing `synthlab` env (recommended):

```bash
conda env create -f synthlab/conda/synthlab.yml
conda activate synthlab
export PYTHONPATH=.
```

Optional dependency install refresh:

```bash
pip install -r medgemma_cup/envs/requirements.txt
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
pytest -q medagent_system/factory/tests
```

Run v1-only regression tests:

```bash
pytest -q medagent_system/factory/tests/test_mvp_smoke.py \
         medagent_system/factory/tests/test_openai_fallback.py \
         medagent_system/factory/tests/test_synthlab_wrapper_unit.py
```

## 4) Local single-patient run

### v1 (MVP)

```bash
export MEDAGENT_PIPELINE_MODE=mvp
python3 medagent_system/runtime/run_mvp.py
```

### v2 (blackboard 6-agent)

```bash
export MEDAGENT_PIPELINE_MODE=v2
python3 medagent_system/runtime/run_mvp.py
```

## 5) BioMCP connectivity check

```bash
python3 medagent_system/runtime/check_biomcp_sdk.py \
  --intent GENERAL_LITERATURE_SUPPORT \
  --query "BRAF melanoma"
```

## 6) Run on cluster (.sbat)

Default script:
- `medagent_system/runtime/run_medagent.sbat`

### v1

```bash
sbatch medagent_system/runtime/run_medagent.sbat --mode v1
```

### v2 + MedGemma + BioMCP SDK + critic + revision loop=2

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v2 \
  --use-medgemma 1 \
  --hf-home $HOME/.cache/huggingface \
  --medgemma-model-id google/medgemma-1.5-4b-it \
  --use-biomcp-sdk 1 \
  --enable-critic 1 \
  --max-supervisor-revisions 2 \
  --max-critic-cycles 1
```

### v2 multi-patient SynthLab job

```bash
sbatch medagent_system/runtime/run_medagent.sbat \
  --mode v2 \
  --use-medgemma 1 \
  --use-biomcp-sdk 1 \
  --synthlab-max-patients 5 \
  --synthlab-modalities fhir,genomics,notes,dicom \
  --synthlab-download-if-missing 1
```

## 7) Outputs and logs

Slurm logs:
- `log/medagent_<JOBID>.out`
- `log/medagent_<JOBID>.err`

Run artifacts:
- `medagent_system/runtime/examples/cluster_runs/<run_id>/final_output.json`
- `medagent_system/runtime/examples/cluster_runs/<run_id>/logs/agent_outputs.jsonl`
- `medagent_system/runtime/examples/cluster_runs/<run_id>/logs/agent_comms.jsonl`

## 8) Optional terminal chat with MedGemma

```bash
./medagent_system/runtime/chat_medgemma.sh
```

## 9) Troubleshooting

- `FileNotFoundError: ~/.cache/synthlab/coherent`:
  use `--synthlab-download-if-missing 1` in `.sbat` or pre-download coherent dataset.
- BioMCP returns empty results:
  run `check_biomcp_sdk.py` first and inspect `debug` block.
- MedGemma fallback used:
  check `final_output.json` at `provenance.blackboard.medgemma_report.notes`.
