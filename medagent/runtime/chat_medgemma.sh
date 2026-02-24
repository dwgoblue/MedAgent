#!/usr/bin/env bash
set -euo pipefail

# Defaults optimized for your cluster cache layout.
export MEDAGENT_USE_MEDGEMMA="${MEDAGENT_USE_MEDGEMMA:-1}"
export MEDAGENT_MEDGEMMA_MODEL_ID="${MEDAGENT_MEDGEMMA_MODEL_ID:-google/medgemma-1.5-4b-it}"
export HF_HOME="${HF_HOME:-/grid/koo/home/dalin/.cache/huggingface}"
export MEDAGENT_MEDGEMMA_LOCAL_ONLY="${MEDAGENT_MEDGEMMA_LOCAL_ONLY:-1}"
export MEDAGENT_MEDGEMMA_DEVICE="${MEDAGENT_MEDGEMMA_DEVICE:-auto}"

python3 medagent/runtime/chat_medgemma.py "$@"
