#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 让 Python 能找到 samples/desktop/x2robot_client
DESKTOP_ROOT = Path(__file__).resolve().parents[1]
if str(DESKTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(DESKTOP_ROOT))

from x2robot.sdk import GripperPosition
from x2robot_client.async_desktop_sdk_client import AsyncDesktopClient


def first_open_index(client, actions: np.ndarray):
    """返回模型动作中第一次明显打开夹爪的位置。"""
    try:
        model_max = float(client._model_gripper_max())
    except Exception:
        model_max = 1.55

    indices = np.flatnonzero(actions[:, -1] >= model_max * 0.5)
    return int(indices[0]) if indices.size else None


def max_pairwise_xyz_mm(chunks):
    """计算任意两次预测轨迹之间的最大 XYZ 点差。"""
    maximum = 0.0

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            a = chunks[i][:, :3]
            b = chunks[j][:, :3]

            length = min(len(a), len(b))
            distances = np.linalg.norm(
                a[:length] - b[:length],
                axis=1,
            )
            maximum = max(maximum, float(distances.max()))

    return maximum * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-address", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--robot-sdk-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--open-right-before-snapshot",
        action="store_true",
        help=(
            "Open the right gripper to 4.5 and wait "
            "before collecting the frozen model input."
        ),
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncDesktopClient(
        model_address=args.model_address,
        model_port=args.port,
        instruction=args.instruction,
        control_mode="end_pose",
        camera_history_k=1,
        camera_capture_hz=20,
        interpolate_multiplier=4,
        robot_sdk_url=args.robot_sdk_url,
        control_hz=80.0,
        inference_workers=1,
    )

    records = []
    left_chunks = []
    right_chunks = []

    try:
        client.start_control()

        if args.open_right_before_snapshot:
            print("Opening right gripper to 4.5...")
            client.robot_controller.right_gripper.set_position(
                GripperPosition(position=4.5)
            )
            time.sleep(2.0)

            measured = (
                client.robot_controller
                .right_gripper
                .get_position()
            )
            physical = (
                float(measured.position)
                if hasattr(measured, "position")
                else float(measured)
            )

            print(
                "Measured right gripper after open:",
                round(physical, 4),
            )

            feedback_max = float(
                getattr(client, "GRIPPER_DATA_MAX", 1.55)
            )
            open_feedback_threshold = 0.75 * feedback_max

            if physical < open_feedback_threshold:
                raise RuntimeError(
                    "右夹爪没有实际打开到安全测试范围；"
                    f"反馈值={physical:.4f}, "
                    f"阈值={open_feedback_threshold:.4f}"
                )

            print(
                "Right gripper is open in feedback range:",
                f"{physical:.4f}/{feedback_max:.4f}",
            )

        # 只采集一次输入。
        # 后面的多次推理完全复用同一状态、同一图片和同一指令。
        inputs = client._collect_sensor_data()

        normalize = getattr(
            client,
            "_normalize_gripper_observation",
            None,
        )
        if normalize is not None:
            normalize(inputs)

        model_right_gripper = float(
            inputs["state"]["follow2_pos"][-1]
        )
        print(
            "Frozen model right-gripper value:",
            round(model_right_gripper, 4),
        )

        print("Collected one frozen input.")
        print(f"Running {args.rounds} identical predictions...")
        print()

        for run_index in range(args.rounds):
            outputs = client.client.predict_sync(inputs)

            left = np.asarray(
                outputs["follow1_pos"],
                dtype=float,
            )
            right = np.asarray(
                outputs["follow2_pos"],
                dtype=float,
            )

            left_chunks.append(left)
            right_chunks.append(right)

            left_delta_mm = (
                left[-1, :3] - left[0, :3]
            ) * 1000.0
            right_delta_mm = (
                right[-1, :3] - right[0, :3]
            ) * 1000.0

            record = {
                "run": run_index + 1,
                "left": {
                    "delta_xyz_mm": left_delta_mm.tolist(),
                    "last_xyz": left[-1, :3].tolist(),
                    "first_open_index": first_open_index(
                        client, left
                    ),
                    "gripper_min": float(left[:, -1].min()),
                    "gripper_max": float(left[:, -1].max()),
                },
                "right": {
                    "delta_xyz_mm": right_delta_mm.tolist(),
                    "last_xyz": right[-1, :3].tolist(),
                    "first_open_index": first_open_index(
                        client, right
                    ),
                    "gripper_min": float(right[:, -1].min()),
                    "gripper_max": float(right[:, -1].max()),
                },
            }
            records.append(record)

            print(
                f"run={run_index + 1:02d}  "
                f"right_delta_mm="
                f"{np.round(right_delta_mm, 2).tolist()}  "
                f"right_open_i="
                f"{record['right']['first_open_index']}"
            )

    finally:
        client.safe_stop()

    right_endpoints = np.asarray(
        [chunk[-1, :3] for chunk in right_chunks]
    )
    left_endpoints = np.asarray(
        [chunk[-1, :3] for chunk in left_chunks]
    )

    summary = {
        "rounds": args.rounds,
        "instruction": args.instruction,
        "right_max_pairwise_xyz_mm": max_pairwise_xyz_mm(
            right_chunks
        ),
        "left_max_pairwise_xyz_mm": max_pairwise_xyz_mm(
            left_chunks
        ),
        "right_endpoint_std_mm": (
            right_endpoints.std(axis=0) * 1000.0
        ).tolist(),
        "left_endpoint_std_mm": (
            left_endpoints.std(axis=0) * 1000.0
        ).tolist(),
        "right_open_indices": [
            record["right"]["first_open_index"]
            for record in records
        ],
        "left_open_indices": [
            record["left"]["first_open_index"]
            for record in records
        ],
        "records": records,
    }

    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== Repeatability summary =====")
    print(
        "right max pairwise xyz mm:",
        round(summary["right_max_pairwise_xyz_mm"], 4),
    )
    print(
        "left max pairwise xyz mm:",
        round(summary["left_max_pairwise_xyz_mm"], 4),
    )
    print(
        "right endpoint std mm:",
        np.round(
            summary["right_endpoint_std_mm"],
            4,
        ).tolist(),
    )
    print(
        "right open indices:",
        summary["right_open_indices"],
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
