#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
CLIENT_ROOT=$(dirname "$SCRIPT_DIR")
WORKSPACE_SRC=$(dirname "$CLIENT_ROOT")

export PYTHONPATH=$CLIENT_ROOT:$WORKSPACE_SRC/robocontrol:$PYTHONPATH

# Configuration
MODEL_ADDRESS=${1:-"39.101.65.229"}
MODEL_PORT=${2:-1175}
INSTRUCTION=${3:-"Pick up the green cup and place it on the tray."}
CONTROL_MODE=${4:-"end_pose"}
INTERPOLATE_MULTIPLIER=${5:-10}
ROBOT_SDK_URL=${6:-"192.168.10.1:50051"}
SMOOTH_CHUNKS=${7:-""}
BLEND_STEPS=${8:-3}

echo "Starting Desktop Client..."
echo "Model: $MODEL_ADDRESS:$MODEL_PORT"
echo "Instruction: $INSTRUCTION"
echo "Control Mode: $CONTROL_MODE"
echo "Smooth Chunks: $SMOOTH_CHUNKS"
echo "Blend Steps: $BLEND_STEPS"

[ -n "$SMOOTH_CHUNKS" ] && SMOOTH_CHUNKS_FLAG="--smooth-chunks" || SMOOTH_CHUNKS_FLAG=""

python3 "$SCRIPT_DIR/start_sdk_desktop.py" \
    --model-address "$MODEL_ADDRESS" \
    --port "$MODEL_PORT" \
    --instruction "$INSTRUCTION" \
    --control-mode "$CONTROL_MODE" \
    --interpolate-multiplier "$INTERPOLATE_MULTIPLIER" \
    --robot_sdk_url "$ROBOT_SDK_URL" \
    --debug-step \
    $SMOOTH_CHUNKS_FLAG \
    --blend-steps "$BLEND_STEPS" 