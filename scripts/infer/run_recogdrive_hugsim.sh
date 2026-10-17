# HUGSIM agent launcher. Point ltf_path (or uniad_path / vad_path) in HUGSIM's configs/sim/<dataset>_base.yaml at this
# file and run its closed_loop.py with the matching --ad; HUGSIM starts it per episode as
# `zsh <this file> <cuda id> <episode output dir>` in its own environment, so export CHECKPOINT, VLM_PATH (and the
# NAVSIM variables of run_recogdrive_closedloop.sh) before closed_loop.py. Plain sh, as zsh runs it.
DRIVER_GPU="$1"
if [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
  case "${CUDA_VISIBLE_DEVICES}" in
    ''|-1) DRIVER_GPU="${CUDA_VISIBLE_DEVICES}" ;;
    *)
      case "$1" in
        ''|*[!0-9]*) echo "HUGSIM device must be a visible CUDA index" >&2; exit 2 ;;
      esac
      # HUGSIM passes a local CUDA index; keep the evaluator's selected physical device.
      DRIVER_GPU=$(printf '%s,' "${CUDA_VISIBLE_DEVICES}" | cut -d ',' -f "$(( $1 + 1 ))")
      if [ -z "${DRIVER_GPU}" ]; then
        echo "HUGSIM device is outside CUDA_VISIBLE_DEVICES" >&2
        exit 2
      fi
      ;;
  esac
fi
CUDA_VISIBLE_DEVICES="${DRIVER_GPU}" SERVE=hugsim OUTPUT_DIR="$2" exec bash "$(dirname "$0")/run_recogdrive_closedloop.sh"
