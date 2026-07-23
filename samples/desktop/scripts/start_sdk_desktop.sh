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
INTERPOLATE_MULTIPLIER=${5:-0}
ROBOT_SDK_URL=${6:-"192.168.10.1:50051"}
MODEL_ACTION_HZ=${7:-30}
EXEC_HZ=${8:-50}
OVERLAP_MODEL_STEPS=${9:-4}

echo "Starting Desktop Client..."
echo "Model: $MODEL_ADDRESS:$MODEL_PORT"
echo "Instruction: $INSTRUCTION"
echo "Control Mode: $CONTROL_MODE"
echo "Model action rate: $MODEL_ACTION_HZ Hz"
echo "Execution rate: $EXEC_HZ Hz"
echo "Chunk overlap: $OVERLAP_MODEL_STEPS model steps"

python3 "$SCRIPT_DIR/start_sdk_desktop.py" \
    --model-address "$MODEL_ADDRESS" \
    --port "$MODEL_PORT" \
    --instruction "$INSTRUCTION" \
    --control-mode "$CONTROL_MODE" \
    --interpolate-multiplier "$INTERPOLATE_MULTIPLIER" \
    --model-action-hz "$MODEL_ACTION_HZ" \
    --exec-hz "$EXEC_HZ" \
    --overlap-model-steps "$OVERLAP_MODEL_STEPS" \
    --robot_sdk_url "$ROBOT_SDK_URL" \
    --debug-step
