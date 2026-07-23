#!/usr/bin/env python3
import argparse

from x2robot_client.desktop_sdk_client import DesktopClient


def main(args):
    client = DesktopClient(
        model_address=args.model_address,
        model_port=args.port,
        instruction=args.instruction,
        control_mode=args.control_mode,
        camera_history_k=args.camera_history_k,
        camera_capture_hz=args.camera_capture_hz,
        interpolate_multiplier=args.interpolate_multiplier,
        model_action_hz=args.model_action_hz,
        exec_hz=args.exec_hz,
        overlap_model_steps=args.overlap_model_steps,
        debug_step=args.debug_step,
        robot_sdk_url=args.robot_sdk_url,
    )

    try:
        client.start_control()
        client.execute_model()
    finally:
        client.safe_stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Start EX001 Dual Arm Robot Client")

    parser.add_argument(
        "--model-address", default="localhost", type=str, help="Model server IP address"
    )

    parser.add_argument("--port", type=int, default=8000, help="Model server port")

    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick up the green cup and place it on the tray.",
        help="Text instruction for the model",
    )

    parser.add_argument(
        "--control-mode",
        type=str,
        default="end_pose",
        choices=["end_pose", "joints"],
        help="Control mode: end_pose or joints",
    )

    parser.add_argument(
        "--camera-history-k",
        type=int,
        default=1,
        help="Number of history frames for camera",
    )

    parser.add_argument(
        "--camera-capture-hz", type=int, default=20, help="Camera capture frequency"
    )

    parser.add_argument(
        "--interpolate-multiplier",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option; ignored by the asynchronous "
            "time-based resampler"
        ),
    )

    parser.add_argument(
        "--model-action-hz",
        type=float,
        default=30.0,
        help="Temporal frequency represented by raw model action steps",
    )

    parser.add_argument(
        "--exec-hz",
        type=float,
        default=50.0,
        help="Target robot action command frequency",
    )

    parser.add_argument(
        "--overlap-model-steps",
        type=int,
        default=4,
        help="Number of raw model steps used for delay-aligned chunk crossfade",
    )

    parser.add_argument(
        "--debug-step",
        action="store_true",
        help="Enable step-by-step debugging for action execution",
    )

    parser.add_argument(
        "--robot_sdk_url",
        default="localhost:50015",
        type=str,
        help="Robot SDK service address (host:port), e.g. localhost:50015",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
