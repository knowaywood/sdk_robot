#!/usr/bin/env python3
import argparse

from x2robot_client.async_desktop_sdk_client import AsyncDesktopClient


def main(args):
    client = AsyncDesktopClient(
        model_address=args.model_address,
        model_port=args.port,
        instruction=args.instruction,
        control_mode=args.control_mode,
        camera_history_k=args.camera_history_k,
        camera_capture_hz=args.camera_capture_hz,
        interpolate_multiplier=args.interpolate_multiplier,
        debug_step=args.debug_step,
        robot_sdk_url=args.robot_sdk_url,
        control_hz=args.control_hz,
        prefetch_margin=args.prefetch_margin,
        blend_duration=args.blend_duration,
        world_lock_duration=args.world_lock_duration,
        replan_interval=args.replan_interval,
        inference_workers=args.inference_workers,
        gripper_close_confirm=args.gripper_close_confirm,
        gripper_release_confirm=args.gripper_release_confirm,
        gripper_reopen_dwell=args.gripper_reopen_dwell,
        gripper_release_travel=args.gripper_release_travel,
        gripper_transition_linear_speed=args.gripper_transition_linear_speed,
        gripper_transition_angular_speed=args.gripper_transition_angular_speed,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
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
        "--instruction", type=str, default="Pick up the green cup and place it on the tray.", help="Text instruction for the model"
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
        default=20,
        help="Interpolate multiplier for action execution",
    )

    parser.add_argument(
        "--control-hz",
        type=float,
        default=140.0,
        help="Robot command frequency; 140 Hz is recommended, maximum is 200 Hz",
    )

    parser.add_argument(
        "--prefetch-margin",
        type=float,
        default=0.1,
        help="Extra action-buffer time kept beyond measured inference latency",
    )

    parser.add_argument(
        "--blend-duration",
        type=float,
        default=0.2,
        help="Arm trajectory blend duration when replacing an action chunk",
    )

    parser.add_argument(
        "--world-lock-duration",
        type=float,
        default=0.25,
        help="Seconds to converge from the handoff state to absolute model targets",
    )

    parser.add_argument(
        "--replan-interval",
        type=float,
        default=1.0,
        help="Minimum seconds to execute a plan before replacing it",
    )

    parser.add_argument(
        "--inference-workers",
        type=int,
        default=1,
        choices=range(1, 5),
        metavar="{1,2,3,4}",
        help="Number of model connections; 1 avoids server-side contention",
    )

    parser.add_argument(
        "--gripper-close-confirm",
        type=float,
        default=0.15,
        help="Seconds a close request must remain stable before grasping",
    )

    parser.add_argument(
        "--gripper-release-confirm",
        type=float,
        default=0.4,
        help="Seconds an opening command must remain stable before release",
    )

    parser.add_argument(
        "--gripper-reopen-dwell",
        type=float,
        default=1.0,
        help="Minimum seconds before a closed gripper may reopen",
    )

    parser.add_argument(
        "--gripper-release-travel",
        type=float,
        default=0.05,
        help="Minimum Cartesian travel after grasp before release is allowed",
    )

    parser.add_argument(
        "--gripper-transition-linear-speed",
        type=float,
        default=0.08,
        help="Maximum Cartesian speed for opening or closing the gripper",
    )

    parser.add_argument(
        "--gripper-transition-angular-speed",
        type=float,
        default=0.6,
        help="Maximum angular speed for opening or closing the gripper",
    )

    parser.add_argument(
        "--max-linear-speed",
        type=float,
        default=0.3,
        help="Maximum Cartesian translation speed in meters per second",
    )

    parser.add_argument(
        "--max-angular-speed",
        type=float,
        default=1.5,
        help="Maximum Cartesian rotation speed in radians per second",
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
