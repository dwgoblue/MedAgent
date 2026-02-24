#!/usr/bin/env bash
set -euo pipefail

# Easy ablation launcher for MedAgent v2 benchmark runs.
# Usage:
#   ./medagent/runtime/run_ablation_grid.sh
#   ./medagent/runtime/run_ablation_grid.sh --dry-run

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

BASE_CMD=(
  sbatch
  medagent/runtime/run_medagent.sbat
  --mode v2
  --use-medgemma 1
  --use-medgemma-image-tensors 1
  --hf-home /grid/koo/home/dalin/.cache/huggingface
  --medgemma-model-id google/medgemma-1.5-4b-it
  --benchmark-outcomes 1
  --benchmark-max-patients 10
  --benchmark-use-biomcp-gate 1
  --kg-backend dashboard
  --synthlab-modalities fhir,genomics,notes,dicom
  --synthlab-download-if-missing 1
)

# Axes:
# A: openai on/off
# B: biomcp sdk on/off
# C: critic cycles (0/1)
OPENAI_OPTS=(0 1)
BIOMCP_OPTS=(0 1)
CRITIC_CYCLES=(0 1)

for use_openai in "${OPENAI_OPTS[@]}"; do
  for use_biomcp in "${BIOMCP_OPTS[@]}"; do
    for critic_cycles in "${CRITIC_CYCLES[@]}"; do
      cmd=("${BASE_CMD[@]}"
        --use-openai "$use_openai"
        --openai-model gpt-5.2
        --use-biomcp-sdk "$use_biomcp"
        --enable-critic 1
        --max-supervisor-revisions 3
        --max-critic-cycles "$critic_cycles"
      )
      echo "== Ablation: openai=$use_openai biomcp_sdk=$use_biomcp critic_cycles=$critic_cycles =="
      printf '%q ' "${cmd[@]}"
      echo
      if [[ "$DRY_RUN" -eq 0 ]]; then
        "${cmd[@]}"
      fi
    done
  done
done
