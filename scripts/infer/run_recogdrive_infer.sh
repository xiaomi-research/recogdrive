#!/usr/bin/env bash
# Trajectories of a trained checkpoint on any registered data source, written to ${OUTPUT_DIR}/predictions.json
# (plus ${OUTPUT_DIR}/vis/*.png for the first VISUALIZE samples). Agent and data overrides go after the script, e.g.
#   CHECKPOINT=.../epoch_0010.ckpt bash scripts/infer/run_recogdrive_infer.sh data_loader=nuscenes nuscenes.root=/data/nuscenes
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
VLM_PATH="${VLM_PATH:-/path/to/internvl3_pretrain_model}"
SPLIT="${SPLIT:-val}"            # train | val | test (NAVSIM: every scene of train_test_split)
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
VISUALIZE="${VISUALIZE:-16}"
GPUS="${GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/infer/$(basename "${CHECKPOINT}" .ckpt)}"

torchrun --standalone --nproc_per_node="${GPUS}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_recogdrive_infer.py" \
  agent=recogdrive_agent \
  agent._target_=recogdrive.adapters.navsim.ReCogDriveAgent \
  agent.vlm_path="${VLM_PATH}" \
  agent.cache_hidden_state=False \
  infer.checkpoint="${CHECKPOINT}" \
  infer.split="${SPLIT}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  infer.visualize="${VISUALIZE}" \
  output_dir="${OUTPUT_DIR}" \
  "$@"
