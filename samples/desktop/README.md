# X2Robot Client SDK (Desktop)

This repository provides a sample client for interacting with the **X2Robot model service** using the **Desktop** SDK workflow.
**This sample is only compatible with Desktop.**

---

## 📦 Requirements

- The SDK must be installed in the same Python virtual environment as described in
  <https://github.com/X-Square-Robot/sdk_robot#virtual-environment-setup>
- Linux
- Network access to the model service

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r samples/desktop/requirements.txt
```

### 2. Run the SDK Client

```bash
bash samples/desktop/scripts/start_sdk_desktop.sh
```

The asynchronous launcher accepts three additional trailing values: model
action rate, robot execution rate, and overlap in raw model steps. For example:

```bash
bash samples/desktop/scripts/start_sdk_desktop.sh \
  "192.168.1.120" 32195 "Pick up the cucumber." "end_pose" \
  0 "192.168.31.251:50051" 30 50 4
```

The fifth positional argument is retained for compatibility with old commands;
the asynchronous time-based resampler no longer uses an integer interpolation
multiplier.

### 🧠 Inference Workflow

- **High-level flow**:
  - Connect to the **model service** (WebSocket) and receive server `metadata`
  - Connect to the **robot SDK service** (`x2://<ROBOT_SDK_URL>`), set SDK work mode and control mode
  - The main thread collects observations and waits for each WebSocket response
  - A separate execution thread continues consuming the current action buffer while inference is running
  - Each response drops the prefix that became stale during its own request, performs a short delay-aligned crossfade, and atomically installs the new buffer
  - Raw model actions are resampled from 30 Hz to the 50 Hz SDK grid; orientation uses Slerp and the gripper uses zero-order hold
- **Per-iteration (code-level) steps**:
  - `_collect_sensor_data()`: read robot state and camera views, build the model input payload
  - `predict_sync()`: send the payload over WebSocket (msgpack) and wait for the model response
  - `_capture_inference_context()`: record request time and committed action sequence
  - `_execute_actions()`: align and install the returned action chunk
  - `_execution_loop()`: send dual-arm SDK commands at the target execution rate
- **About “press Enter to continue”**:
  - The Desktop launcher passes `--debug-step`, but `DesktopClient` currently doesn’t pause on Enter. The client runs continuously by default (for step-by-step behavior, see the `Quanta_X1` sample or add an interactive pause in `DesktopClient`).

### ⚙️ Default Configuration

The script uses the following default parameters:
| Parameter | Description | Default |
|  ----  | ----  | ----  |
|`MODEL_ADDRESS` | Model service IP address | `39.101.65.229` |
|`MODEL_PORT` | Model service port | `1175` |
|`INSTRUCTION` | Initial instruction sent to the model |`Pick up the green cup and place it on the tray`|
|`CONTROL_MODE` | Control mode for the robot | `end_pose` |
|`INTERPOLATE_MULTIPLIER` | Deprecated compatibility value (ignored) |`0` |
|`ROBOT_SDK_URL` | Robot SDK address | `192.168.10.1:50051` |
|`MODEL_ACTION_HZ` | Raw model action rate | `30` |
|`EXEC_HZ` | Target SDK command rate | `50` |
|`OVERLAP_MODEL_STEPS` | Delay-aligned overlap in raw model steps | `4` |

### 📌 Notes

- Ensure the model service is running and reachable before starting the client.
- `Action buffer underrun` means inference is slower than the remaining buffered future.
- `Installed action chunk` reports latency, committed/skipped steps, overlap, and new buffer length.
- This is phase-one delay-aware asynchronous chunk scheduling, not flow-level RTC inpainting.
- This sample is intended for Desktop inference and SDK validation.
- Logs and responses will be printed directly to the terminal.

### 📄 License

This project is provided for internal testing and SDK integration purposes.
