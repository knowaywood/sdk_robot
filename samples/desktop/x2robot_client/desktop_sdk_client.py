import datetime
import logging
import os
import time
from typing import Dict, List

import cv2
import numpy as np
from scipy.spatial import transform
from x2robot import Robot, connect
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
from x2robot_client.sdk_utils import compressed_image_to_numpy, interpolate_trajectory


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
        interpolate_multiplier: int = 1,
        debug_step: bool = False,
        robot_sdk_url: str = "",
        save_debug_plot: bool = False,
        smooth_chunks: bool = False,
        blend_steps: int = 3,
    ):
        self.control_mode = control_mode
        self.camera_history_k = camera_history_k
        self.camera_capture_hz = camera_capture_hz
        self.interpolate_multiplier = interpolate_multiplier
        self.debug_step = debug_step
        self.save_debug_plot = save_debug_plot
        self.robot_controller = None
        self.last_arm_l_pos = None
        self.last_arm_r_pos = None
        self.plot_dir = None
        self.robot_sdk_url = robot_sdk_url
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
        super().__init__(model_address, model_port, instruction, max_retries, smooth_chunks, blend_steps)

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

    def safe_stop(self):
        """Stop control and sensors."""
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

    def _execute_actions(self, outputs: Dict):
        """Execute actions."""
        arm1_actions = outputs.get("follow1_pos")
        arm2_actions = outputs.get("follow2_pos")

        if arm1_actions is None or arm2_actions is None:
            return

        if self.interpolate_multiplier > 1:
            if self.last_arm_l_pos is not None:
                arm1_actions = [self.last_arm_l_pos] + arm1_actions
            if self.last_arm_r_pos is not None:
                arm2_actions = [self.last_arm_r_pos] + arm2_actions

        arm1_actions = interpolate_trajectory(
            arm1_actions, self.interpolate_multiplier, self.control_mode
        )
        arm2_actions = interpolate_trajectory(
            arm2_actions, self.interpolate_multiplier, self.control_mode
        )

        if self.control_mode == "end_pose":
            logger.info(f"Executing end_pose actions: {len(arm1_actions)} steps")

            for i in range(len(arm1_actions)):
                # arm1
                a1 = arm1_actions[i]  # [x,y,z, r,p,y, gripper]
                p1 = list(a1[:3])
                e1 = list(a1[3:6])
                q1 = transform.Rotation.from_euler("xyz", e1).as_quat().tolist()
                g1 = float(a1[6])

                # arm2
                a2 = arm2_actions[i]
                p2 = list(a2[:3])
                e2 = list(a2[3:6])
                q2 = transform.Rotation.from_euler("xyz", e2).as_quat().tolist()
                g2 = float(a2[6])
                g1 = max(0.0, min(4.5, g1))
                g2 = max(0.0, min(4.5, g2))

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
                self.robot_controller.left_gripper.set_position(
                    GripperPosition(position=g1)
                )
                self.robot_controller.right_gripper.set_position(
                    GripperPosition(position=g2)
                )
                time.sleep(0.005)
        else:  # joints mode

            for i in range(len(arm1_actions)):
                # arm1
                a1 = arm1_actions[i]
                j1 = list(a1[:-1])
                g1 = float(a1[-1])

                # arm2
                a2 = arm2_actions[i]
                j2 = list(a2[:-1])
                g2 = float(a2[-1])
                g1 = max(0.0, min(4.5, g1))
                g2 = max(0.0, min(4.5, g2))

                j1_cmd = JointPositions(positions=j1)
                j2_cmd = JointPositions(positions=j2)

                self.robot_controller.left_arm.set_joint_positions(j1_cmd)
                self.robot_controller.right_arm.set_joint_positions(j2_cmd)
                self.robot_controller.left_gripper.set_position(
                    GripperPosition(position=g1)
                )
                self.robot_controller.right_gripper.set_position(
                    GripperPosition(position=g2)
                )
                time.sleep(0.005)
