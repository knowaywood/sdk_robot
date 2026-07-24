#!/bin/bash
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
CLIENT_ROOT=$(dirname "$SCRIPT_DIR")
WORKSPACE_SRC=$(dirname "$CLIENT_ROOT")
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd)
LOG_FILE=${DESKTOP_ASYNC_LOG_FILE:-"/tmp/desktop_async.log"}

export PYTHONPATH=$CLIENT_ROOT:$WORKSPACE_SRC/robocontrol:$PYTHONPATH

# Configuration
MODEL_ADDRESS=${1:-"39.101.65.229"}
MODEL_PORT=${2:-1175}
INSTRUCTION=${3:-"Pick up the green cup and place it on the tray."}
CONTROL_MODE=${4:-"end_pose"}
INTERPOLATE_MULTIPLIER=${5:-10}
ROBOT_SDK_URL=${6:-"192.168.10.1:50051"}

echo "Starting asynchronous Desktop Client..."
echo "Workspace: $REPO_ROOT"
echo "Implementation: x2robot_client.async_desktop_sdk_client.AsyncDesktopClient"
echo "Log: $LOG_FILE"
echo "Model: $MODEL_ADDRESS:$MODEL_PORT"
echo "Instruction: $INSTRUCTION"
echo "Control Mode: $CONTROL_MODE"

PYTHONUNBUFFERED=1 python3 "$SCRIPT_DIR/start_sdk_desktop.py" \
    --model-address "$MODEL_ADDRESS" \
    --port "$MODEL_PORT" \
    --instruction "$INSTRUCTION" \
    --control-mode "$CONTROL_MODE" \
    --interpolate-multiplier "$INTERPOLATE_MULTIPLIER" \
    --robot_sdk_url "$ROBOT_SDK_URL" \
    "${@:7}" \
    --debug-step 2>&1 | tee "$LOG_FILE"
