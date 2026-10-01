# HUGSIM agent launcher. Point ltf_path (or uniad_path / vad_path) in HUGSIM's configs/sim/<dataset>_base.yaml at this
# file and run its closed_loop.py with the matching --ad; HUGSIM starts it per episode as
# `zsh <this file> <cuda id> <episode output dir>` in its own environment, so export CHECKPOINT, VLM_PATH (and the
# NAVSIM variables of run_recogdrive_closedloop.sh) before closed_loop.py. Plain sh, as zsh runs it.
CUDA_VISIBLE_DEVICES="$1" SERVE=hugsim OUTPUT_DIR="$2" exec bash "$(dirname "$0")/run_recogdrive_closedloop.sh"
