#!/usr/bin/env bash
# Run MedAgent pipelines on a personal cluster or bare-metal machine (no job scheduler).
# Same options as run_medagent.sbat, plus --cpus, --gpus, --ngpus for local control.
# Usage:
#   ./medagent/runtime/run_medagent_local.sh --mode v2 --use-medgemma 1 ...
#   ./medagent/runtime/run_medagent_local.sh --ngpus 2 --mode v2 ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPUS="8"
GPUS=""
NGPUS=""

# Parse local-only options and pass the rest to run_medagent.sbat.
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) CPUS="${2:?missing value for --cpus}"; shift 2 ;;
    --gpus) GPUS="${2:?missing value for --gpus}"; shift 2 ;;
    --ngpus) NGPUS="${2:?missing value for --ngpus}"; shift 2 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

# Simulate SLURM environment so run_medagent.sbat behaves correctly.
export SLURM_CPUS_PER_TASK="$CPUS"
export SLURM_JOB_ID="local"
export SLURM_SUBMIT_DIR="$REPO_ROOT"

# Optional: restrict to specific GPU(s). MedGemma uses one GPU; use --ngpus 1 to avoid
# touching others, or --gpus 0,1,2,3 to pin to those devices.
if [[ -n "$NGPUS" ]]; then
  # Use first N GPUs (0,1,...,N-1).
  GPUS=""
  for (( i=0; i < NGPUS; i++ )); do
    [[ -n "$GPUS" ]] && GPUS="$GPUS,"
    GPUS="${GPUS}$i"
  done
fi
if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

cd "$REPO_ROOT"
exec bash "$SCRIPT_DIR/run_medagent.sbat" "${PASSTHROUGH[@]}"
