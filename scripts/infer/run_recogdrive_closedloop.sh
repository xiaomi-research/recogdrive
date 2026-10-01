#!/usr/bin/env bash
# Serves a trained checkpoint to a closed-loop simulator (recogdrive.closedloop): SERVE=alpasim | neuroncap.
# The agent config is the training one; overrides go after the script, e.g.
#   CHECKPOINT=.../epoch_0010.ckpt SERVE=neuroncap PORT=9000 bash scripts/infer/run_recogdrive_closedloop.sh agent.cameras=null
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAVSIM_TREE="${NAVSIM_TREE:-1.1}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/path/to/NAVSIM/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/path/to/NAVSIM/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/navsim${NAVSIM_TREE}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/NAVSIM/dataset}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a trained .ckpt}"
SERVE="${SERVE:?set SERVE to alpasim or neuroncap}"
VLM_PATH="${VLM_PATH:-/path/to/internvl3_pretrain_model}"
PORT="${PORT:-null}"
HOST="${HOST:-127.0.0.1}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_recogdrive_infer.py" \
  agent=recogdrive_agent \
  agent._target_=recogdrive.adapters.navsim.ReCogDriveAgent \
  agent.vlm_path="${VLM_PATH}" \
  agent.cache_hidden_state=False \
  infer.checkpoint="${CHECKPOINT}" \
  infer.serve="${SERVE}" \
  infer.port="${PORT}" \
  infer.host="${HOST}" \
  output_dir="${NAVSIM_EXP_ROOT}/closedloop/${SERVE}" \
  "$@"
