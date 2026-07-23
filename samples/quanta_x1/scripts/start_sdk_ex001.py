#!/usr/bin/env python3
import argparse

from x2robot_client.ex001_sdk_client import EX001SDKClient


def main(args):
    client = EX001SDKClient(
        model_address=args.model_address,
        model_port=args.port,
        instruction=args.instruction,
        control_mode=args.control_mode,
        camera_history_k=args.camera_history_k,
        camera_capture_hz=args.camera_capture_hz,
        interpolate_multiplier=args.interpolate_multiplier,
        use_car_pose_odom=args.use_car_pose_odom,
        debug_step=args.debug_step,
        robot_sdk_url=args.robot_sdk_url,
        smooth_chunks=args.smooth_chunks,
        blend_steps=args.blend_steps,
    )

    try:
        client.start_control()
        client.execute_model()
    finally:
        client.safe_stop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-address", default="localhost", type=str, help="Model server IP address"
    )

    parser.add_argument("--port", type=int, default=8000, help="Model server port")

    parser.add_argument(
        "--instruction", type=str, default="", help="Text instruction for the model"
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
        default=10,
        help="Interpolate multiplier for action execution",
    )

    parser.add_argument(
        "--robot_sdk_url",
        default="localhost:50015",
        type=str,
        help="Robot SDK service address (host:port), e.g. localhost:50015",
    )

    parser.add_argument(
        "--debug-step",
        action="store_true",
        help="Enable step-by-step debugging for action execution",
    )

    parser.add_argument(
        "--use-car-pose-odom",
        action="store_true",
        help="Use car pose odom for chassis control",
    )

    parser.add_argument(
        "--smooth-chunks",
        action="store_true",
        help="Enable smooth chunk boundary blending",
    )

    parser.add_argument(
        "--blend-steps",
        type=int,
        default=3,
        help="Number of initial steps to blend when smooth-chunks is enabled",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
