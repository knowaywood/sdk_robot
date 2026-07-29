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
        debug_step=args.debug_step,
        robot_sdk_url=args.robot_sdk_url,
        smooth_chunks=args.smooth_chunks,
        blend_steps=args.blend_steps,
        rtc_enabled=args.rtc_enabled,
        rtc_delay_steps=max(0, args.rtc_delay_ms // 33),
        rtc_overlap_steps=args.rtc_overlap_steps,
        rtc_blend=args.rtc_blend,
        async_mode=args.async_mode,
    )

    try:
        client.start_control()
        client.execute_model()
    finally:
        client.safe_stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Start Desktop Robot Client")

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

    parser.add_argument(
        "--rtc-enabled",
        action="store_true",
        help="Enable server-side RTC (three-zone denoising blending on server)",
    )

    parser.add_argument(
        "--rtc-delay-ms",
        type=int,
        default=100,
        help="RTC delay in milliseconds (D=0/100/200)",
    )

    parser.add_argument(
        "--rtc-overlap-steps",
        type=int,
        default=3,
        help="Number of overlap steps for RTC blending (M)",
    )

    parser.add_argument(
        "--rtc-blend",
        action="store_true",
        help="Enable client-side three-zone RTC blending",
    )

    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Enable async pipeline: inference runs in background thread, "
             "execution picks up latest chunk without blocking",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
