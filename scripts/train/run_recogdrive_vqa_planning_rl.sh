#!/usr/bin/env bash
# Alternate VQA LM updates with planning GRPO updates through one trainable VLM.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGE=rl bash "${SCRIPT_DIR}/run_recogdrive_vqa_planning_il.sh" "$@"
