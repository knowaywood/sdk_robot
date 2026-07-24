#!/usr/bin/env python3

import argparse
import base64
import json
import sys
import time
from pathlib import Path

# Make samples/desktop importable regardless of the current working directory.
DESKTOP_ROOT = Path(__file__).resolve().parents[1]
if str(DESKTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(DESKTOP_ROOT))

import numpy as np

from x2robot_client.async_desktop_sdk_client import AsyncDesktopClient


def jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value


def action_summary(outputs):
    result = {}

    for key in ("follow1_pos", "follow2_pos"):
        actions = outputs.get(key)

        if actions is None:
            result[key] = {"missing": True}
            continue

        array = np.asarray(actions, dtype=float)

        result[key] = {
            "shape": list(array.shape),
            "first": array[0].tolist(),
            "last": array[-1].tolist(),
            "xyz_min": array[:, :3].min(axis=0).tolist(),
            "xyz_max": array[:, :3].max(axis=0).tolist(),
            "xyz_displacement_from_first": np.linalg.norm(
                array[:, :3] - array[0, :3],
                axis=1,
            ).tolist(),
            "gripper_min": float(array[:, -1].min()),
            "gripper_max": float(array[:, -1].max()),
            "gripper_values": array[:, -1].tolist(),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-address", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--robot-sdk-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncDesktopClient(
        model_address=args.model_address,
        model_port=args.port,
        instruction=args.instruction,
        control_mode="end_pose",
        camera_history_k=1,
        camera_capture_hz=20,
        interpolate_multiplier=5,
        robot_sdk_url=args.robot_sdk_url,
        control_hz=80.0,
        inference_workers=1,
    )

    try:
        # 只初始化连接并采集一次，不调用 execute_model，
        # 不发送机械臂或夹爪动作。
        client.start_control()

        inputs = client._collect_sensor_data()
        client._normalize_gripper_observation(inputs)

        for view_name, encoded_image in inputs.get("views", {}).items():
            if not encoded_image:
                print(f"Missing image: {view_name}")
                continue

            image_bytes = base64.b64decode(encoded_image)
            image_path = output_dir / f"{view_name}.jpg"
            image_path.write_bytes(image_bytes)
            print(f"Saved: {image_path}")

        input_metadata = {
            "instruction": jsonify(inputs.get("instruction")),
            "state": jsonify(inputs.get("state")),
            "view_names": list(inputs.get("views", {}).keys()),
        }

        with (output_dir / "model_input.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                input_metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        started = time.monotonic()
        outputs = client.client.predict_sync(inputs)
        elapsed = time.monotonic() - started

        with (output_dir / "raw_model_output.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                jsonify(outputs),
                file,
                ensure_ascii=False,
                indent=2,
            )

        summary = {
            "round_trip_seconds": elapsed,
            "instruction": args.instruction,
            "actions": action_summary(outputs),
        }

        with (output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved diagnostic snapshot to: {output_dir}")

    finally:
        client.safe_stop()


if __name__ == "__main__":
    main()
