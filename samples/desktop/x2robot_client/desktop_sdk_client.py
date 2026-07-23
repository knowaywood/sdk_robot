import datetime
import logging
import os
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np
from scipy.spatial import transform
from x2robot import connect
from x2robot.geometry_msgs import Point, Pose, Quaternion
from x2robot.sdk import (
    GripperPosition,
    JointPositions,
    ManipulatorControlMode,
    ManipulatorControlModeParam,
    RobotModeParam,
    RobotWorkMode,
)
from x2robot.sensor_msgs import CompressedImage
from x2robot_client.robot_client_base import RobotClientBase
from x2robot_client.sdk_utils import (
    blend_action_pair,
    compressed_image_to_numpy,
    resample_trajectory,
)


class SDKClientLogger:
    """Logger class for EX001 SDK Client."""

    def __init__(self, name: str = "ex001_sdk_client", level: int = logging.INFO):
        """
        Initialize logger.

        Args:
            name: Logger name
            level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)

            self.logger.addHandler(handler)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


# Create logger instance
logger = SDKClientLogger(__name__)


class DesktopClient(RobotClientBase):
    """Desktop Robot Client."""

    def __init__(
        self,
        model_address: str,
        model_port: int,
        instruction: str = "",
        control_mode: str = "end_pose",  # "end_pose" or "joints"
        camera_history_k: int = 1,
        camera_capture_hz: int = 20,
        max_retries: int = 3,
        interpolate_multiplier: int = 0,
        model_action_hz: float = 30.0,
        exec_hz: float = 50.0,
        overlap_model_steps: int = 4,
        debug_step: bool = False,
        robot_sdk_url: str = "",
        save_debug_plot: bool = False,
    ):
        if model_action_hz <= 0 or exec_hz <= 0:
            raise ValueError("model_action_hz and exec_hz must be positive")
        if overlap_model_steps < 0:
            raise ValueError("overlap_model_steps must be non-negative")

        self.control_mode = control_mode
        self.camera_history_k = camera_history_k
        self.camera_capture_hz = camera_capture_hz
        # Kept only so existing launch commands do not fail. Time-based
        # resampling replaces the old integer interpolation multiplier.
        self.interpolate_multiplier = interpolate_multiplier
        self.model_action_hz = float(model_action_hz)
        self.exec_hz = float(exec_hz)
        self.overlap_model_steps = int(overlap_model_steps)
        self.debug_step = debug_step
        self.save_debug_plot = save_debug_plot
        self.robot_controller = None
        self.last_arm_l_pos = None
        self.last_arm_r_pos = None
        self.plot_dir = None
        self.robot_sdk_url = robot_sdk_url
        self._action_buffer_l = []
        self._action_buffer_r = []
        self._step_idx = 0
        self._exec_seq = 0
        self._buffer_generation = 0
        self._request_seq = 0
        self._last_sent_l = None
        self._last_sent_r = None
        self._exec_lock = threading.Lock()
        self._stop_execution = threading.Event()
        self._exec_thread = None
        self._underrun_warned = False
        self._last_overrun_warning_at = 0.0
        self._exec_rate_window_started_at = time.monotonic()
        self._exec_rate_window_count = 0

        if self.interpolate_multiplier not in (0, 1):
            logger.warning(
                "--interpolate-multiplier is deprecated and ignored by the "
                "asynchronous scheduler; use --model-action-hz and --exec-hz"
            )
        if self.save_debug_plot:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use absolute path so user can find it reliably (based on current working directory).
            self.plot_dir = os.path.abspath(f"debug_plots_{timestamp}")
            os.makedirs(self.plot_dir, exist_ok=True)
            self.plot_counter = 0
            cwd = os.getcwd()
            logger.info(
                f"[DEBUG PLOT] Saving action_chunk plots to: {self.plot_dir} (cwd when started: {cwd})"
            )
        super().__init__(model_address, model_port, instruction, max_retries)

    def _init_robot_controller(self):
        """Initialize robot controller using robocontrol."""
        self.robot_controller = connect(f"x2://{self.robot_sdk_url}")
        self.robot_controller.system.set_work_mode(
            RobotModeParam(mode=RobotWorkMode.SDK)
        )

        if self.robot_controller is None:
            raise RuntimeError("Failed to get robot 'desktop'")

        mode_map = {
            "end_pose": ManipulatorControlModeParam(
                mode=ManipulatorControlMode.MANIPULATOR_END_POSE
            ),
            "joints": ManipulatorControlModeParam(
                mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS
            ),
        }
        if self.control_mode not in mode_map:
            raise ValueError(f"Unsupported control_mode: {self.control_mode}")

        self.robot_controller.robot_control.set_manipulator_control_mode(
            mode_map[self.control_mode]
        )

    def start_control(self):
        """Start control and sensors."""
        super().start_control()
        if self._exec_thread is not None and self._exec_thread.is_alive():
            return
        self._stop_execution.clear()
        self._exec_rate_window_started_at = time.monotonic()
        self._exec_rate_window_count = 0
        self._exec_thread = threading.Thread(
            target=self._execution_loop,
            name="desktop-action-executor",
            daemon=True,
        )
        self._exec_thread.start()
        logger.info(
            f"Asynchronous action executor started at {self.exec_hz:.1f} Hz "
            f"(model action rate {self.model_action_hz:.1f} Hz)"
        )

    def safe_stop(self):
        """Stop control and sensors."""
        stop_event = getattr(self, "_stop_execution", None)
        if stop_event is not None:
            stop_event.set()
        exec_thread = getattr(self, "_exec_thread", None)
        if (
            exec_thread is not None
            and exec_thread.is_alive()
            and threading.current_thread() is not exec_thread
        ):
            exec_thread.join(timeout=2.0)
        super().safe_stop()

    @staticmethod
    def _compress_image(image_np):
        """Compress image to JPEG base64 string."""
        if image_np is None:
            return None

        # Ensure image_np is a numpy array
        if not isinstance(image_np, np.ndarray):
            logger.warning(f"Image is not a numpy array, got type: {type(image_np)}")
            return None

        success, encoded = cv2.imencode(".jpg", image_np)
        if not success:
            return None
        import base64

        base64_str = base64.b64encode(encoded).decode("utf-8")
        return base64_str

    def _get_camera_image(self, camera_name):
        """Get compressed image from robot controller directly."""
        img = None

        try:
            if camera_name == "left_camera":
                img_raw = self.robot_controller.left_arm_camera.get_raw_image()

            elif camera_name == "right_camera":
                img_raw = self.robot_controller.right_arm_camera.get_raw_image()

            elif camera_name == "head_camera":
                img_raw = self.robot_controller.head_camera.get_rgb_image()
                # print(f"img_raw: {img_raw}")
            else:
                return None

            # Convert CompressedImage to numpy array if needed
            if isinstance(img_raw, CompressedImage):
                img = compressed_image_to_numpy(img_raw)
            elif isinstance(img_raw, np.ndarray):
                img = img_raw
            else:
                logger.warning(
                    f"Unexpected image type from {camera_name}: {type(img_raw)}"
                )
                return None

            return self._compress_image(img)
        except Exception as e:
            logger.warning(f"Failed to get image from {camera_name}: {e}")
            return None

    def _collect_sensor_data(self) -> Dict:
        """Collect sensor data."""
        # 1. Get Arm Data
        if self.control_mode == "end_pose":
            # Get end pose (x, y, z, qx, qy, qz, qw)
            l_pose = self.robot_controller.left_arm.get_end_pose()
            r_pose = self.robot_controller.right_arm.get_end_pose()

            # Convert quaternion to euler for model input if needed, or keep as is.
            # Example DesktopACTClient converts back to euler: [x, y, z, r, p, y] + [gripper]
            # We follow the example.
            def _process_pose(pose, gripper_val):
                # pose: [x, y, z, qx, qy, qz, qw]
                pos = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                quat = [
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ]
                euler = transform.Rotation.from_quat(quat).as_euler("xyz").tolist()
                return pos + euler + [gripper_val]

        else:  # joints
            # Get joint positions
            l_pose = self.robot_controller.left_arm.get_joint_states().position
            r_pose = self.robot_controller.right_arm.get_joint_states().position

            def _process_pose(pose, gripper_val):
                return list(pose) + [gripper_val]

        # Get Gripper State
        l_gripper_obj = self.robot_controller.left_gripper.get_position()
        r_gripper_obj = self.robot_controller.right_gripper.get_position()

        # Extract numeric position value from GripperPosition object
        l_gripper = (
            l_gripper_obj.position
            if hasattr(l_gripper_obj, "position")
            else float(l_gripper_obj)
        )
        r_gripper = (
            r_gripper_obj.position
            if hasattr(r_gripper_obj, "position")
            else float(r_gripper_obj)
        )

        arm_l_pos = _process_pose(l_pose, l_gripper)
        arm_r_pos = _process_pose(r_pose, r_gripper)

        self.last_arm_l_pos = arm_l_pos
        self.last_arm_r_pos = arm_r_pos
        arm_l_cur = self.robot_controller.left_arm.get_joint_states().effort
        arm_r_cur = self.robot_controller.right_arm.get_joint_states().effort

        # Direct fetch from robot controller
        cam_left = self._get_camera_image("left_camera")
        cam_right = self._get_camera_image("right_camera")
        cam_front = self._get_camera_image(
            "head_camera"
        )  # Assuming head_camera is front

        # Get End Effort Wrench (force + torque)
        arm_l_wrench = self.robot_controller.left_arm.get_wrench_ext_local()
        arm_l_end_effort_force = [
            arm_l_wrench.wrench.force.x,
            arm_l_wrench.wrench.force.y,
            arm_l_wrench.wrench.force.z,
        ]
        arm_l_end_effort_torque = [
            arm_l_wrench.wrench.torque.x,
            arm_l_wrench.wrench.torque.y,
            arm_l_wrench.wrench.torque.z,
        ]
        arm_r_wrench = self.robot_controller.right_arm.get_wrench_ext_local()
        arm_r_end_effort_force = [
            arm_r_wrench.wrench.force.x,
            arm_r_wrench.wrench.force.y,
            arm_r_wrench.wrench.force.z,
        ]
        arm_r_end_effort_torque = [
            arm_r_wrench.wrench.torque.x,
            arm_r_wrench.wrench.torque.y,
            arm_r_wrench.wrench.torque.z,
        ]

        inputs = {
            "state": {
                "follow1_pos": np.array(arm_l_pos, dtype=np.float32),
                "follow2_pos": np.array(arm_r_pos, dtype=np.float32),
                "follow1_joints_cur": np.array(arm_l_cur, dtype=np.float32),
                "follow2_joints_cur": np.array(arm_r_cur, dtype=np.float32),
                "follow1_gripper": np.array(l_gripper, dtype=np.float32),
                "follow2_gripper": np.array(r_gripper, dtype=np.float32),
                "follow1_end_effort_force": np.array(
                    arm_l_end_effort_force, dtype=np.float32
                ),
                "follow1_end_effort_torque": np.array(
                    arm_l_end_effort_torque, dtype=np.float32
                ),
                "follow2_end_effort_force": np.array(
                    arm_r_end_effort_force, dtype=np.float32
                ),
                "follow2_end_effort_torque": np.array(
                    arm_r_end_effort_torque, dtype=np.float32
                ),
            },
            "views": {
                "camera_left": cam_left,
                "camera_front": cam_front,
                "camera_right": cam_right,
            },
            "instruction": np.array([self.instruction], dtype=np.object_),
        }

        return inputs

    def _capture_inference_context(self) -> Dict[str, Any]:
        """Record the execution point represented by the current observation."""
        with self._exec_lock:
            self._request_seq += 1
            context = {
                "request_id": self._request_seq,
                "started_at": time.monotonic(),
                "exec_seq": self._exec_seq,
                "buffer_generation": self._buffer_generation,
                "had_active_buffer": self._step_idx < len(self._action_buffer_l),
            }
        return context

    def _send_single_action(self, a1, a2):
        """Send one synchronized dual-arm command."""
        if self.control_mode == "end_pose":
            p1 = list(a1[:3])
            q1 = transform.Rotation.from_euler(
                "xyz", list(a1[3:6])
            ).as_quat()
            g1 = max(0.0, min(4.5, float(a1[-1])))

            p2 = list(a2[:3])
            q2 = transform.Rotation.from_euler(
                "xyz", list(a2[3:6])
            ).as_quat()
            g2 = max(0.0, min(4.5, float(a2[-1])))

            self.robot_controller.left_arm.set_end_pose(
                Pose(
                    position=Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
                    orientation=Quaternion(
                        x=float(q1[0]),
                        y=float(q1[1]),
                        z=float(q1[2]),
                        w=float(q1[3]),
                    ),
                )
            )
            self.robot_controller.right_arm.set_end_pose(
                Pose(
                    position=Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2])),
                    orientation=Quaternion(
                        x=float(q2[0]),
                        y=float(q2[1]),
                        z=float(q2[2]),
                        w=float(q2[3]),
                    ),
                )
            )
        else:
            self.robot_controller.left_arm.set_joint_positions(
                JointPositions(positions=list(a1[:-1]))
            )
            self.robot_controller.right_arm.set_joint_positions(
                JointPositions(positions=list(a2[:-1]))
            )
            g1 = max(0.0, min(4.5, float(a1[-1])))
            g2 = max(0.0, min(4.5, float(a2[-1])))

        self.robot_controller.left_gripper.set_position(
            GripperPosition(position=g1)
        )
        self.robot_controller.right_gripper.set_position(
            GripperPosition(position=g2)
        )

    def _execution_loop(self):
        """Consume the active buffer independently of WebSocket inference."""
        interval = 1.0 / self.exec_hz
        next_tick = time.monotonic()

        while not self._stop_execution.is_set():
            next_tick += interval
            a_l = a_r = None

            with self._exec_lock:
                if self._step_idx < len(self._action_buffer_l):
                    a_l = self._action_buffer_l[self._step_idx]
                    a_r = self._action_buffer_r[self._step_idx]
                    self._step_idx += 1
                    # Count the command as committed while holding the same lock
                    # used by response alignment.
                    self._exec_seq += 1
                    self._underrun_warned = False
                elif self._action_buffer_l and not self._underrun_warned:
                    logger.warning(
                        "Action buffer underrun; holding the last commanded pose"
                    )
                    self._underrun_warned = True

            if a_l is not None:
                try:
                    self._send_single_action(a_l, a_r)
                    achieved_rate = None
                    with self._exec_lock:
                        self._last_sent_l = list(a_l)
                        self._last_sent_r = list(a_r)
                        command_time = time.monotonic()
                        if self._exec_rate_window_count == 0:
                            self._exec_rate_window_started_at = command_time
                        self._exec_rate_window_count += 1
                        rate_elapsed = (
                            command_time - self._exec_rate_window_started_at
                        )
                        if rate_elapsed >= 5.0:
                            achieved_rate = (
                                self._exec_rate_window_count / rate_elapsed
                            )
                            self._exec_rate_window_started_at = command_time
                            self._exec_rate_window_count = 0
                    if achieved_rate is not None:
                        logger.info(
                            f"Achieved action command rate: {achieved_rate:.1f} Hz "
                            f"(target {self.exec_hz:.1f} Hz)"
                        )
                except Exception as exc:
                    logger.error(f"Action execution failed: {exc}")
                    self.action_terminator = True
                    self._stop_execution.set()
                    break

            remaining = next_tick - time.monotonic()
            if remaining > 0:
                self._stop_execution.wait(remaining)
            elif remaining < -interval:
                now = time.monotonic()
                if now - self._last_overrun_warning_at >= 1.0:
                    logger.warning(
                        f"Execution loop overrun by {-remaining * 1000.0:.1f} ms"
                    )
                    self._last_overrun_warning_at = now
                next_tick = now

    def _resample_outputs(self, outputs: Dict):
        arm_l = outputs.get("follow1_pos")
        arm_r = outputs.get("follow2_pos")
        if arm_l is None or arm_r is None:
            return None, None

        arm_l = list(arm_l)
        arm_r = list(arm_r)
        if len(arm_l) != len(arm_r):
            logger.warning(
                f"Arm chunk length mismatch: left={len(arm_l)}, right={len(arm_r)}"
            )
            chunk_length = min(len(arm_l), len(arm_r))
            arm_l = arm_l[:chunk_length]
            arm_r = arm_r[:chunk_length]
        if not arm_l:
            return None, None

        return (
            resample_trajectory(
                arm_l, self.model_action_hz, self.exec_hz, self.control_mode
            ),
            resample_trajectory(
                arm_r, self.model_action_hz, self.exec_hz, self.control_mode
            ),
        )

    def _execute_actions(
        self, outputs: Dict, inference_context: Dict[str, Any] = None
    ):
        """Align a returned chunk to now and atomically install its future."""
        exec_l, exec_r = self._resample_outputs(outputs)
        if exec_l is None or exec_r is None:
            logger.warning("Model response did not contain a usable dual-arm chunk")
            return

        now = time.monotonic()
        context = inference_context or {}
        request_id = context.get("request_id", -1)
        wall_delay = max(0.0, now - context.get("started_at", now))

        with self._exec_lock:
            committed_delay = max(
                0, self._exec_seq - int(context.get("exec_seq", self._exec_seq))
            )
            had_active_buffer = bool(context.get("had_active_buffer", False))

            # The initial request has no old trajectory executing, so its
            # prefix is not discarded merely because inference took time.
            delay_steps = committed_delay if had_active_buffer else 0
            expected_delay = int(round(wall_delay * self.exec_hz))

            if delay_steps >= len(exec_l):
                logger.warning(
                    f"Discarding stale response request={request_id}: "
                    f"delay={delay_steps} steps, chunk={len(exec_l)} steps"
                )
                return

            aligned_l = exec_l[delay_steps:]
            aligned_r = exec_r[delay_steps:]
            old_remaining_l = self._action_buffer_l[self._step_idx :]
            old_remaining_r = self._action_buffer_r[self._step_idx :]

            overlap_exec_steps = int(
                round(self.overlap_model_steps * self.exec_hz / self.model_action_hz)
            )
            overlap = min(
                overlap_exec_steps,
                len(old_remaining_l),
                len(old_remaining_r),
                len(aligned_l),
                len(aligned_r),
            )

            stitched_l = []
            stitched_r = []
            for index in range(overlap):
                alpha = float(index + 1) / float(overlap)
                stitched_l.append(
                    blend_action_pair(
                        old_remaining_l[index],
                        aligned_l[index],
                        alpha,
                        self.control_mode,
                    )
                )
                stitched_r.append(
                    blend_action_pair(
                        old_remaining_r[index],
                        aligned_r[index],
                        alpha,
                        self.control_mode,
                    )
                )

            stitched_l.extend(aligned_l[overlap:])
            stitched_r.extend(aligned_r[overlap:])

            self._action_buffer_l = stitched_l
            self._action_buffer_r = stitched_r
            # The stale prefix was removed above; the new buffer always starts
            # at index zero. _step_idx never leaks across chunk generations.
            self._step_idx = 0
            self._buffer_generation += 1
            generation = self._buffer_generation
            self._underrun_warned = False

        logger.info(
            f"Installed action chunk generation={generation}, request={request_id}, "
            f"latency={wall_delay * 1000.0:.1f} ms, "
            f"committed_delay={committed_delay}, expected_delay={expected_delay}, "
            f"skipped={delay_steps}, overlap={overlap}, "
            f"buffer={len(stitched_l)} steps"
        )
