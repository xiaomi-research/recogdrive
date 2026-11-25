#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
GPU="${1:?Pass the number of GPUs}"
MODEL="${MODEL:?Set MODEL to the checkpoint or Hugging Face model ID}"
DATA_JSON="${DATA_JSON:-data/drivebench-test-final.json}"
OUTPUT_DIR="${OUTPUT_DIR:-res/llava-1.5-7b}"

# Define corresponding output names and corruption values in order.
corruptions=(
    ""              # clean
    "NoImage"       # noimage
    "BitError"      # biterror
    "CameraCrash"   # cameracrash
    "Fog"           # fog
    "H256ABRCompression"  # h256
    "LowLight"      # lowlight
    "Rain"          # rain
    "Snow"          # snow
    "Brightness"    # bright
    "ColorQuant"    # colorquant
    "FrameLost"     # framelost
    "LensObstacleCorruption"  # lens
    "MotionBlur"    # motion
    "Saturate"      # saturate
    "ZoomBlur"      # zoom
    "WaterSplashCorruption"   # water
)

outputs=(
    "clean"
    "noimage"
    "biterror"
    "cameracrash"
    "fog"
    "h256"
    "lowlight"
    "rain"
    "snow"
    "bright"
    "colorquant"
    "framelost"
    "lens"
    "motion"
    "saturate"
    "zoom"
    "water"
)

# Loop over the arrays and run the commands.
for i in "${!outputs[@]}"; do
    python inference/llava1.5.py \
        --model "$MODEL" \
        --data "$DATA_JSON" \
        --output "$OUTPUT_DIR/${outputs[i]}" \
        --system_prompt prompt.txt \
        --num_processes "${GPU}" \
        --max_model_len 4096 \
        --corruption "${corruptions[i]}"
done
