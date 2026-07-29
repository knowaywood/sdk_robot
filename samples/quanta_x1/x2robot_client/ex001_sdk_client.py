import logging
import subprocess
import time
from typing import Dict, List, Optional
import os
import datetime
import cv2
import numpy as np
import matplotlib
import typer
# Set backend to Agg to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import transform
from x2robot import Robot, connect
from x2robot.geometry_msgs import Point, Pose, Quaternion
from x2robot.sdk import (
    ChassisControlMode,
    ChassisControlModeParam,
    ChassisPosition,
    ChassisPositionList,
    ChassisVelocity,
    CoordinateSystemMode,
    CoordinateSystemModeParam,
    GripperPosition,
    HeadPose,
    JointPositions,
    LiftPosition,
    ManipulatorControlMode,
    ManipulatorControlModeParam,
    NavigationMode,
    NavigationModeParam,
    RobotModeParam,
    RobotWorkMode,
    SaveMapParam,
)
from x2robot.sensor_msgs import CompressedImage
from x2robot_client.robot_client_base import RobotClientBase
from x2robot_client.sdk_utils import (
    compressed_image_to_numpy,
    interpolate_trajectory,
    restore_gripper_from_raw,
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


class EX001SDKClient(RobotClientBase):
    """EX001 Robot Client."""

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
        use_car_pose_odom: bool = False,
        robot_sdk_url: str = "",
        save_debug_plot: bool = False,
        smooth_chunks: bool = False,
        blend_steps: int = 3,
        rtc_enabled: bool = False,
        rtc_delay_steps: int = 0,
        rtc_overlap_steps: int = 3,
        rtc_blend: bool = False,
    ):
        self.control_mode = control_mode
        self.camera_history_k = camera_history_k
        self.camera_capture_hz = camera_capture_hz
        self.interpolate_multiplier = interpolate_multiplier
        self.debug_step = debug_step
        self.use_car_pose_odom = use_car_pose_odom
        self.save_debug_plot = save_debug_plot
        self.robot_controller = None
        # self.last_car_pose_odom_actions = None
        self.last_velocity_actions = None
        self.last_sensor_state = None  # Store last sensor state for visualization
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
        super().__init__(
            model_address, model_port, instruction, max_retries,
            smooth_chunks, blend_steps,
            rtc_enabled, rtc_delay_steps, rtc_overlap_steps, rtc_blend,
        )

    def _init_robot_controller(self):
        """Initialize robot controller using robocontrol."""
        self.robot_controller = connect(f"x2://{self.robot_sdk_url}")
        self.robot_controller.system.set_work_mode(
            RobotModeParam(mode=RobotWorkMode.SDK)
        )

        if self.robot_controller is None:
            raise RuntimeError("Failed to get robot 'ex001'")

        # Set arm modes
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

        result = self.robot_controller.navigation.set_navigation_mode(
            NavigationModeParam(mode=NavigationMode.BUILT_IN_NAVIGATION)
        )
        if not result.is_success:
            raise RuntimeError(f"Failed to set navigation mode: {result.message}")

        # Initialize chassis mode to relative
        self.robot_controller.chassis.set_control_mode(
            ChassisControlModeParam(mode=ChassisControlMode.RELATIVE)
        )

    def _move_around_to_build_map(self):
        """Move around to build map."""
        self.robot_controller.chassis.set_control_mode(
            ChassisControlModeParam(mode=ChassisControlMode.VELOCITY)
        )
        for i in range(100):
            cur_velocity = ChassisVelocity(vel_x=-0.2, vel_y=0.0, vel_yaw=0.0)
            self.robot_controller.chassis.set_velocity(cur_velocity)
            time.sleep(0.01)
        for i in range(100):    
            cur_velocity = ChassisVelocity(vel_x=0.2, vel_y=0.0, vel_yaw=0.0)
            self.robot_controller.chassis.set_velocity(cur_velocity)
            time.sleep(0.01)

        time.sleep(0.5)
        for i in range(800):
            cur_velocity = ChassisVelocity(vel_x=0.0, vel_y=0.0, vel_yaw=0.4)
            self.robot_controller.chassis.set_velocity(cur_velocity)
            time.sleep(0.01)
        time.sleep(0.5)
        for i in range(800):
            cur_velocity = ChassisVelocity(vel_x=0.0, vel_y=0.0, vel_yaw=-0.4)
            self.robot_controller.chassis.set_velocity(cur_velocity)
            time.sleep(0.01)
        for i in range(10):
            cur_velocity = ChassisVelocity(vel_x=0.0, vel_y=0.0, vel_yaw=0.0)
            self.robot_controller.chassis.set_velocity(cur_velocity)
            time.sleep(0.01)
        time.sleep(1.0)
        self.robot_controller.chassis.set_control_mode(
            ChassisControlModeParam(mode=ChassisControlMode.RELATIVE)
        )

    def start_control(self):
        """Start control and sensors."""
        # Set virtual zero pose for chassis at start
        if self.robot_controller:
            confirm = typer.confirm("Do you want to build new map?", default=True)
            if confirm:
                self.robot_controller.navigation.start_mapping()
                self._move_around_to_build_map()
                time.sleep(3.0)
                self.robot_controller.navigation.stop_mapping(SaveMapParam(map_name="test"))
        
            self.robot_controller.navigation.start_localization(
                SaveMapParam(map_name="test")
            )
            time.sleep(3.0)
            current_pose = self.robot_controller.chassis.get_global_position()
            self.robot_controller.chassis.set_virtual_zero_point(current_pose)
            logger.info(f"Chassis virtual zero set to: {current_pose}")

        if self.robot_controller.chassis:
            # set chassis coordinate system mode to odometry
            coord_system_mode = CoordinateSystemModeParam(
                coordinate_system_mode=CoordinateSystemMode.ODOMETRY
            )
            self.robot_controller.chassis.set_trajectory_coord_system_mode(
                coord_system_mode
            )
            self.robot_controller.chassis.reset_navigation_chunk_id()
            logger.info(f"Chassis coordinate system mode set to: {coord_system_mode}")

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

    def _get_camera_image(self, camera_name, resize_height=None):
        """Get compressed image from robot controller directly."""

        img = None

        try:
            if camera_name == "left_camera":
                img_raw = self.robot_controller.left_arm_camera.get_raw_image()

            elif camera_name == "right_camera":
                img_raw = self.robot_controller.right_arm_camera.get_raw_image()

            elif camera_name == "head_camera":
                img_raw = self.robot_controller.head_camera.get_rgb_image()
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

            if img is not None and resize_height is not None:
                h, w = img.shape[:2]
                if h > resize_height:
                    scale = resize_height / h
                    new_w = int(w * scale)
                    new_h = int(resize_height)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return self._compress_image(img)
        except Exception as e:
            logger.warning(f"Failed to get image from {camera_name}: {e}")
            return None

    @staticmethod
    def velocity_to_pose(vx_body, vy_body, vyaw, dt, start_pose):
        """Converts body frame velocity to global frame pose."""
        x, y, theta = start_pose
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        dx_global = (vx_body * cos_theta - vy_body * sin_theta) * dt
        dy_global = (vx_body * sin_theta + vy_body * cos_theta) * dt
        dtheta = vyaw * dt
        x_new, y_new = x + dx_global, y + dy_global
        theta_new = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi
        return np.array([x_new, y_new, theta_new])

    def _collect_sensor_data(self) -> Dict:
        """Collect sensor data."""
        if self.debug_step:
            input(
                f"\n[DEBUG STEP] Press Enter to collect data and request next chunk..."
            )

        # 1. Get Cartesian Pose (Always collected for follow1_pos standard)
        l_end_pose = self.robot_controller.left_arm.get_end_pose()
        # [x, y, z, qx, qy, qz, qw] list
        r_end_pose = self.robot_controller.right_arm.get_end_pose()

        def _process_end_pose(pose, gripper_val):
            # Handle PoseStamped object (has pose.pose.position and pose.pose.orientation)
            if hasattr(pose, "pose"):
                # PoseStamped structure
                pos = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                quat = [
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ]
            elif hasattr(pose, "position"):
                # Direct Pose structure
                pos = [pose.position.x, pose.position.y, pose.position.z]
                quat = [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            else:
                # Fallback: assume it's a list [x, y, z, qx, qy, qz, qw]
                pos = pose[:3]
                quat = pose[3:]

            # Convert to numpy array for easier manipulation
            quat = np.array(quat, dtype=np.float64)
            
            # Check for invalid values (NaN, Inf)
            if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
                # Use identity quaternion [0, 0, 0, 1] as fallback
                quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            else:
                # Check for zero norm quaternion and handle it
                quat_norm = np.linalg.norm(quat)
                if quat_norm < 1e-6:  # Very small norm, treat as zero
                    # Use identity quaternion [0, 0, 0, 1] as fallback
                    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                else:
                    # Normalize the quaternion to ensure it's valid
                    quat = quat / quat_norm

            # Convert to list format expected by scipy (scipy can handle numpy arrays, but list is safer)
            quat_list = quat.tolist()
            
            # Try to convert quaternion to Euler angles, with fallback to identity quaternion
            try:
                euler = transform.Rotation.from_quat(quat_list).as_euler("xyz").tolist()
            except ValueError:
                # If conversion still fails, use identity quaternion [0, 0, 0, 1] -> [0, 0, 0] Euler
                logging.warning(f"Invalid quaternion detected: {quat_list}, using identity rotation")
                euler = [0.0, 0.0, 0.0]
            return pos + euler + [gripper_val]

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

        arm_l_pos = _process_end_pose(l_end_pose, l_gripper)
        arm_r_pos = _process_end_pose(r_end_pose, r_gripper)

        # 2. Get Joint Positions (Always collected for follow1_joints standard)
        l_joints = self.robot_controller.left_arm.get_joint_states().position
        r_joints = self.robot_controller.right_arm.get_joint_states().position

        arm_l_joints = list(l_joints)
        arm_r_joints = list(r_joints)

        # 3. Get Joint Efforts/Currents
        l_arm_joint_state = next(self.robot_controller.left_arm.get_joint_states_stream())
        r_arm_joint_state = next(self.robot_controller.right_arm.get_joint_states_stream())
        arm_l_cur = list(l_arm_joint_state.effort)
        arm_r_cur = list(r_arm_joint_state.effort)
        # Append gripper effort from left_ee/right_ee
        l_gripper_joint_state = next(self.robot_controller.left_gripper.get_joint_states_stream())
        l_gripper_effort = l_gripper_joint_state.effort[0]
        r_gripper_joint_state = next(self.robot_controller.right_gripper.get_joint_states_stream())
        r_gripper_effort = r_gripper_joint_state.effort[0]
        arm_l_cur.append(float(l_gripper_effort) if l_gripper_effort is not None else 0.0)
        arm_r_cur.append(float(r_gripper_effort) if r_gripper_effort is not None else 0.0)
        
        # 3.5 Get Joint Velocities (dev = velocity/deviation) (6 joints + 1 gripper = 7)
        arm_l_vel = list(self.robot_controller.left_arm.get_joint_states().velocity)
        arm_r_vel = list(self.robot_controller.right_arm.get_joint_states().velocity)
        # Gripper velocity not available in robocontrol, append 0.0
        arm_l_vel.append(0.0)
        arm_r_vel.append(0.0)
        
        # 4. Get Head, Lift, Chassis Data
        head_state = (
            self.robot_controller.head.get_pose()
        )  # Robot interface returns [pitch, yaw]
        # Flip to [yaw, pitch] for API consistency
        head_pos = [head_state.yaw, head_state.pitch]

        lift_state_obj = (
            self.robot_controller.lift.get_lift_position()
        )  # LiftPosition object
        # Extract numeric position value from LiftPosition object
        lift_state = (
            lift_state_obj.position
            if hasattr(lift_state_obj, "position")
            else float(lift_state_obj)
        )

        # Extract numeric values from ChassisPosition object
        chassis_state_obj = (
            self.robot_controller.chassis.get_relative_position()
        )  # ChassisPosition object
        if (
            hasattr(chassis_state_obj, "x")
            and hasattr(chassis_state_obj, "y")
            and hasattr(chassis_state_obj, "yaw")
        ):
            chassis_state = [
                chassis_state_obj.x,
                chassis_state_obj.y,
                chassis_state_obj.yaw,
            ]
        else:
            # Fallback: try to convert directly
            chassis_state = (
                list(chassis_state_obj)
                if hasattr(chassis_state_obj, "__iter__")
                else [float(chassis_state_obj)]
            )

        # Try to get velocity
        chassis_vel_state = [0.0, 0.0, 0.0]
        if (
            self.last_velocity_actions is not None
            and len(self.last_velocity_actions) >= 2
        ):
            try:
                # Use the last two points to calculate velocity
                p_curr = np.array(self.last_velocity_actions[-1])
                p_prev = np.array(self.last_velocity_actions[-2])

                dt = 1.0 / self.camera_capture_hz

                dx = p_curr[0] - p_prev[0]
                dy = p_curr[1] - p_prev[1]
                dtheta = p_curr[2] - p_prev[2]

                # Unwrap angle diff
                dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

                # Transform to body frame of p_prev
                theta = p_prev[2]
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                vx = (dx * cos_t + dy * sin_t) / dt
                vy = (-dx * sin_t + dy * cos_t) / dt
                w = dtheta / dt

                chassis_vel_state = [vx, vy, w]
                logger.info(f"Calculated velocity state: {chassis_vel_state}")
            except Exception as e:
                logger.warning(f"Failed to calculate velocity: {e}")

        # 5. Get Camera Data
        cam_left = self._get_camera_image("left_camera")
        cam_right = self._get_camera_image("right_camera")
        cam_front = self._get_camera_image("head_camera", resize_height=480)

        # Construct inputs following API.md
        inputs = {
            "state": {
                "follow1_pos": np.array(arm_l_pos, dtype=np.float32),
                "follow2_pos": np.array(arm_r_pos, dtype=np.float32),
                "follow1_joints": np.array(arm_l_joints, dtype=np.float32),
                "follow2_joints": np.array(arm_r_joints, dtype=np.float32),
                "follow1_joints_cur": np.array(arm_l_cur, dtype=np.float32),
                "follow2_joints_cur": np.array(arm_r_cur, dtype=np.float32),
                "head_pos": np.array(head_pos, dtype=np.float32),
                "lift": np.array([lift_state], dtype=np.float32),
                "car_pose": np.array(chassis_state, dtype=np.float32),
                "velocity_decomposed": np.array(chassis_vel_state, dtype=np.float32),
                "velocity_decomposed_odom": np.array(
                    chassis_vel_state, dtype=np.float32
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

    def _wait_for_chassis_target(self, target_pose, timeout=5.0):
        """Wait for chassis to reach target pose."""
        t0 = time.time()
        POS_TOL = 0.05  # 5 cm
        ANG_TOL = 0.1  # ~5.7 degrees

        dist = 999.0
        yaw_diff = 999.0

        while time.time() - t0 < timeout:
            try:
                curr_pose_obj = self.robot_controller.chassis.get_relative_position()
                # Extract numeric values from ChassisPosition object
                if (
                    hasattr(curr_pose_obj, "x")
                    and hasattr(curr_pose_obj, "y")
                    and hasattr(curr_pose_obj, "yaw")
                ):
                    curr_pose = np.array(
                        [curr_pose_obj.x, curr_pose_obj.y, curr_pose_obj.yaw],
                        dtype=float,
                    )
                else:
                    curr_pose = np.array(curr_pose_obj, dtype=float)
                if len(curr_pose) < 3:
                    time.sleep(0.02)
                    continue

                dist = np.linalg.norm(target_pose[:2] - curr_pose[:2])
                yaw_diff = abs(
                    (target_pose[2] - curr_pose[2] + np.pi) % (2 * np.pi) - np.pi
                )

                if dist < POS_TOL and yaw_diff < ANG_TOL:
                    return
            except Exception as e:
                logger.warning(f"Error in wait_for_chassis: {e}")

            time.sleep(0.02)

        logger.warning(
            f"Timeout waiting for chassis. Dist={dist:.3f}, YawDiff={yaw_diff:.3f}"
        )


    @staticmethod
    def _extract_lift_action(
        lift_actions: np.ndarray, step_idx: int
    ) -> Optional[float]:
        """
        Extract lift action from lift_actions array.
        """
        idx = min(step_idx, len(lift_actions) - 1)
        lift_action = lift_actions[idx]
        if isinstance(lift_action, list):
            lift_action = lift_action[0]
        return lift_action
    
    def _decode_arm_actions(self, outputs: Dict):
        """Decode arm actions based on control mode."""
        if self.control_mode == "end_pose":
            arm1_actions = self._decode_ndarray(outputs.get("follow1_pos"))
            arm2_actions = self._decode_ndarray(outputs.get("follow2_pos"))
        else:  # joints
            arm1_actions = self._decode_ndarray(outputs.get("follow1_joints"))
            arm2_actions = self._decode_ndarray(outputs.get("follow2_joints"))
            # Fallback if joints specific key not present but model outputs to pos key (legacy compat)
            if arm1_actions is None:
                arm1_actions = self._decode_ndarray(outputs.get("follow1_pos"))
            if arm2_actions is None:
                arm2_actions = self._decode_ndarray(outputs.get("follow2_pos"))
        return arm1_actions, arm2_actions

    @staticmethod
    def _decode_ndarray(data):
        """Decode msgpack-numpy dictionary to numpy array."""
        if data is None:
            return None
        if isinstance(data, dict) and (b"__ndarray__" in data or "__ndarray__" in data):
            try:
                # Handle both bytes and string keys just in case
                dtype_key = b"dtype" if b"dtype" in data else "dtype"
                shape_key = b"shape" if b"shape" in data else "shape"
                data_key = b"data" if b"data" in data else "data"

                dtype = np.dtype(data[dtype_key])
                shape = data[shape_key]
                return np.frombuffer(data[data_key], dtype=dtype).reshape(shape)
            except Exception as e:
                logger.warning(f"Failed to decode ndarray: {e}")
                return data
        return data


    def _decode_auxiliary_actions(self, outputs: Dict):
        """Decode head, lift, chassis and odom actions."""
        head_actions = self._decode_ndarray(outputs.get("head_pos"))
        lift_actions = self._decode_ndarray(outputs.get("lift"))
        
        chassis_vel_actions = self._decode_ndarray(outputs.get("velocity_decomposed"))
        if chassis_vel_actions is None:
            chassis_vel_actions = self._decode_ndarray(outputs.get("velocity_decomposed_odom"))
            print("!!!velocity_decomposed_odom:",chassis_vel_actions)
        
        car_pose_odom_actions = self._decode_ndarray(outputs.get("car_pose_odom"))
        return head_actions, lift_actions, chassis_vel_actions, car_pose_odom_actions


    def _interpolate_arm_actions(self, actions):
        """Interpolate arm actions trajectory."""
        if actions is not None:
            return interpolate_trajectory(
                actions, self.interpolate_multiplier, self.control_mode
            )
        return None

    def _prepare_chassis_trajectory(self, chassis_vel_actions, car_pose_odom_actions, dt_model):
        """Compute chassis pose trajectory from actions."""
        pose_actions = None
        if (
            self.use_car_pose_odom
            and car_pose_odom_actions is not None
            and len(car_pose_odom_actions) > 0
        ):
            pose_actions = interpolate_trajectory(
                car_pose_odom_actions, self.interpolate_multiplier, mode="joints"
            )
        else:
            if self.use_car_pose_odom:
                logger.warning(
                    "use_car_pose_odom enabled but car_pose_odom not provided; falling back to velocity integration"
                )
            if chassis_vel_actions is not None and len(chassis_vel_actions) > 0:
                start_pose_obj = self.robot_controller.chassis.get_relative_position()
                # Extract numeric values from ChassisPosition object
                if (
                    hasattr(start_pose_obj, "x")
                    and hasattr(start_pose_obj, "y")
                    and hasattr(start_pose_obj, "yaw")
                ):
                    start_pose = np.array(
                        [start_pose_obj.x, start_pose_obj.y, start_pose_obj.yaw],
                        dtype=float,
                    )
                else:
                    start_pose = np.array(start_pose_obj, dtype=float)
                logger.info(f"start_pose is: {start_pose}")
                pose_actions = []
                pose_curr = start_pose.copy()
                for vel in chassis_vel_actions:
                    assert (
                        len(vel) >= 3
                    ), f"chassis_vel_actions element is not len>=3: {vel}"
                    pose_curr = self.velocity_to_pose(vel[0], vel[1], vel[2], dt_model, pose_curr)
                    pose_actions.append(pose_curr.copy())
                logger.info(f"pose_actions before interpolate_trajectory are: {pose_actions}")   
                pose_actions = interpolate_trajectory(pose_actions, self.interpolate_multiplier, mode="joints")
                self.last_velocity_actions = pose_actions

        return pose_actions
    
    def _control_arms(self, i, arm1_actions, arm2_actions, steps):
        """Control arms."""
        if self.control_mode == "end_pose":
            # follow1_pos: [x, y, z, r, p, y, gripper]
            a1 = arm1_actions[i]
            p1 = list(a1[:3])
            e1 = list(a1[3:6])
            q1 = transform.Rotation.from_euler("xyz", e1).as_quat().tolist()
            g1 = float(a1[6])
            a2 = arm2_actions[i]
            p2 = list(a2[:3])
            e2 = list(a2[3:6])
            q2 = transform.Rotation.from_euler("xyz", e2).as_quat().tolist()
            g2 = float(a2[6])
            if g1 < 0.0:
                # logger.warning(f"left gripper value is out of range: {g1}")
                g1 = 0.0
            if g1 > 4.5:
                # logger.warning(f"left gripper value is out of range: {g1}")
                g1 = 4.5
            if g2 < 0.0:
                # logger.warning(f"right gripper value is out of range: {g2}")
                g2 = 0.0
            if g2 > 4.5:
                # logger.warning(f"right gripper value is out of range: {g2}")
                g2 = 4.5
            logger.warning(f"left gripper value is set to {g1}")
            logger.warning(f"right gripper value is set to {g2}")
            # Convert to Python floats for protobuf serialization
            # self.robot_controller.left_arm.send_cmd_end_pose(p1 + q1)
            # self.robot_controller.right_arm.send_cmd_end_pose(p2 + q2)
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
        else:
            a1 = arm1_actions[i]
            a2 = arm2_actions[i]
            # Assume last is gripper if length > joint_dof (7)
            if len(a1) > 7:
                j1 = list(a1[:-1])
                g1 = float(a1[-1])
            else:
                j1 = list(a1)
                g1 = None  # Should we hold last gripper?
            if len(a2) > 7:
                j2 = list(a2[:-1])
                g2 = float(a2[-1])
            else:
                j2 = list(a2)
                g2 = None
            j1_cmd = JointPositions(positions=j1)
            j2_cmd = JointPositions(positions=j2)
            self.robot_controller.left_arm.set_joint_positions(j1_cmd)
            self.robot_controller.right_arm.set_joint_positions(j2_cmd)
            if g1 is not None:
                self.robot_controller.left_gripper.set_position(
                    GripperPosition(position=g1)
                )
            if g2 is not None:
                self.robot_controller.right_gripper.set_position(
                    GripperPosition(position=g2)
                )

    def _control_head(self, i, head_actions, steps):
        """Send control commands to head."""
        if head_actions is not None:
            # Handle different lengths (chunk vs single vs hold)
            idx = min(i, len(head_actions) - 1)
            h_cmd = head_actions[idx]
            assert (
                len(h_cmd) >= 2
            ), f"head_actions[idx] is not a list of length 2: {h_cmd}"
            logger.info(f"set head pose to yaw={h_cmd[0]}, pitch={h_cmd[1]}")
            if h_cmd[1] > 0.9:
                logger.warning(f"pitch is too large: {h_cmd[1]}")
                h_cmd[1] = 0.9
            if h_cmd[1] < -0.06:
                logger.warning(f"pitch is too small: {h_cmd[1]}")
                h_cmd[1] = -0.06
            if h_cmd[0] > 1.2:
                logger.warning(f"yaw is too large: {h_cmd[0]}")
                h_cmd[0] = 1.2
            if h_cmd[0] < -1.2:
                logger.warning(f"yaw is too small: {h_cmd[0]}")
                h_cmd[0] = -1.2
            self.robot_controller.head.set_pose(
                HeadPose(yaw=h_cmd[0], pitch=h_cmd[1])
            )
            if i % 10 == 0:
                logger.info(f"Step {i}/{steps}: Head Cmd=[{h_cmd[1]:.3f}, {h_cmd[0]:.3f}]")
        else:
            if i == 0:
                logger.warning(
                    f"HEAD: head_actions is None, skipping head control"
                )

    def _control_lift(self, i, lift_actions, steps):
        """Send control commands to lift."""
        if lift_actions is not None:
            lift_action = self._extract_lift_action(lift_actions, i)
            if lift_action is None:
                if i == 0:
                    logger.warning(
                        f"LIFT: _extract_lift_action returned None at step 0"
                    )
                return
            self.robot_controller.lift.set_lift_position(
                LiftPosition(position=float(lift_action))
            )
            if i % 10 == 0:
                logger.info(f"Step {i}/{steps}: Lift Cmd={lift_action:.3f}")
        else:
            if i == 0:
                logger.warning(
                    f"LIFT: lift_actions is None, skippinremote_controlg lift control"
                )

    def _control_chassis(self, i, pose_actions, steps):
        """Send control commands to chassis."""
        if pose_actions is not None and len(pose_actions) > 0:
            idx = min(i, len(pose_actions) - 1)
            target_pose = pose_actions[idx]
            if i % 10 == 0:
                logger.info(f"Step {i}/{steps}: Chassis Cmd={target_pose}")
            if i == 0:
                logger.info(
                    f"CHASSIS: Sending first command to robot_controller.chassis.send_relative_pose_to_navigation({target_pose})"
                )
          
            chassis_position = ChassisPosition(
                    x=float(target_pose[0]),
                    y=float(target_pose[1]),
                    yaw=float(target_pose[2]),
                )
        
            pose_list = ChassisPositionList()
            pose_list.positions.append(chassis_position)
            self.robot_controller.chassis.send_relative_pose_to_navigation(pose_list)
        else:
            if i == 0:
                logger.warning(
                    f"CHASSIS: pose_actions is None or empty, skipping chassis control"
                )

    def _execute_control_loop(self, steps, arm1_actions, arm2_actions, head_actions, lift_actions, pose_actions):
        """Execute the main control loop for all components."""
        for i in range(steps):
            if not self.remote_control:
                self._control_arms(i, arm1_actions, arm2_actions, steps)
                self._control_head(i, head_actions, steps)
                self._control_lift(i, lift_actions, steps)
                self._control_chassis(i, pose_actions, steps)
                time.sleep(0.005)  # Loop rate
            else:
                time.sleep(0.005)  # Loop rate

    def _visualize_actions(self, arm1_actions, arm2_actions, head_actions, lift_actions, chassis_actions, chassis_vel_actions=None, raw_arm1=None, raw_arm2=None):
        """Visualize actions."""
        if not self.save_debug_plot:
            return
        if self.plot_dir is None:
            logger.warning("[DEBUG PLOT] Skipped: plot_dir is None (save_debug_plot was False at init?).")
            return
        if self.last_sensor_state is None:
            logger.warning("[DEBUG PLOT] Skipped: last_sensor_state is None (save_debug_plot was False at init?).")
            return

        try:
            # Prepare data
            current_state = self.last_sensor_state

            # Helper to pad or reshape data
            def get_current(key, expected_dim):
                val = current_state.get(key)
                if val is None:
                    return np.zeros(expected_dim)
                return np.array(val).flatten()

            # --- Extract Current State ---
            curr_arm1 = get_current("follow1_pos", 7)
            curr_arm2 = get_current("follow2_pos", 7)
            curr_head = get_current("head_pos", 2)
            curr_lift = get_current("lift", 1)
            curr_chassis_vel = get_current("velocity_decomposed_odom", 3)
            curr_chassis_pos = get_current("car_pose", 3)
            
            def _to_2d_action(act, expected_dim: int) -> Optional[np.ndarray]:
                """
                Convert action to shape (T, expected_dim).
                - If act is None: return None
                - If 1D:
                  - expected_dim==1: treat as (T,1)
                  - size==expected_dim: treat as (1,expected_dim)
                  - size%expected_dim==0: reshape to (-1,expected_dim)
                - If 2D: return as-is (and we'll pad/trim columns later)
                """
                if act is None:
                    return None
                arr = np.asarray(act)
                if arr.ndim == 0:
                    # Scalar
                    if expected_dim != 1:
                        return np.full((1, expected_dim), np.nan)
                    return arr.reshape(-1, 1)
                if arr.ndim == 1:
                    if expected_dim == 1:
                        return arr.reshape(-1, 1)
                    if arr.size == expected_dim:
                        return arr.reshape(1, expected_dim)
                    if arr.size % expected_dim == 0:
                        return arr.reshape(-1, expected_dim)
                    # Unknown 1D layout, keep as single row and pad later
                    out = np.full((1, expected_dim), np.nan)
                    out[0, : min(expected_dim, arr.size)] = arr[: min(expected_dim, arr.size)]
                    return out
                if arr.ndim >= 2:
                    return arr.reshape(arr.shape[0], -1)
                return None
                
            def _pad_or_trim_cols(arr2d: Optional[np.ndarray], expected_dim: int) -> Optional[np.ndarray]:
                if arr2d is None:
                    return None
                if arr2d.shape[1] == expected_dim:
                    return arr2d
                out = np.full((arr2d.shape[0], expected_dim), np.nan, dtype=float)
                take = min(expected_dim, arr2d.shape[1])
                out[:, :take] = arr2d[:, :take]
                return out
            def _drop_action0(arr2d: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if arr2d is None:
                    return None
                if arr2d.shape[0] <= 1:
                    return arr2d[:0]  # empty, so plot will just show current pose
                return arr2d[1:]
            pred_arm1_full = _pad_or_trim_cols(_to_2d_action(arm1_actions, 7), 7)
            pred_arm2_full = _pad_or_trim_cols(_to_2d_action(arm2_actions, 7), 7)
            pred_head_full = _pad_or_trim_cols(_to_2d_action(head_actions, 2), 2)
            pred_lift_full = _pad_or_trim_cols(_to_2d_action(lift_actions, 1), 1)
            pred_chassis_vel_full = _pad_or_trim_cols(_to_2d_action(chassis_vel_actions, 3), 3)
            pred_chassis_pos_full = _pad_or_trim_cols(_to_2d_action(chassis_actions, 3), 3)
            
            pred_arm1 = pred_arm1_full
            pred_arm2 = pred_arm2_full
            pred_head = _drop_action0(pred_head_full)
            pred_lift = _drop_action0(pred_lift_full)
            pred_chassis_vel = pred_chassis_vel_full
            pred_chassis_pos = pred_chassis_pos_full
            
            def _raw_on_interp_axis(raw_2d, interp_len):
                if raw_2d is None or raw_2d.shape[0] == 0:
                    return None, None
                R = raw_2d.shape[0]
                M = self.interpolate_multiplier
                j_max = min(R, (interp_len - 1) // M) if interp_len > 0 else 0
                if j_max <= 0:
                    return None, None
                x_raw = np.array([j * M for j in range(1, j_max + 1)], dtype=float)
                y_raw = raw_2d[1:j_max + 1]  # skip prepended agent_pos at index 0
                return x_raw, y_raw
            raw_arm1_2d = _pad_or_trim_cols(_to_2d_action(raw_arm1, 7), 7) if raw_arm1 is not None else None
            raw_arm2_2d = _pad_or_trim_cols(_to_2d_action(raw_arm2, 7), 7) if raw_arm2 is not None else None
            L1 = pred_arm1.shape[0] if pred_arm1 is not None and pred_arm1.ndim == 2 else 0
            L2 = pred_arm2.shape[0] if pred_arm2 is not None and pred_arm2.ndim == 2 else 0
            x_raw_1, y_raw_1 = _raw_on_interp_axis(raw_arm1_2d, L1) if raw_arm1_2d is not None else (None, None)
            x_raw_2, y_raw_2 = _raw_on_interp_axis(raw_arm2_2d, L2) if raw_arm2_2d is not None else (None, None)
            
            dim = 23
            fig, axes = plt.subplots(dim, 1, figsize=(12, 4 * dim))
            
            action_labels = [
                'Left Arm X', 'Left Arm Y', 'Left Arm Z',  # 0-2
                'Left Arm Roll', 'Left Arm Pitch', 'Left Arm Yaw',  # 3-5
                'Left Gripper',  # 6
                'Right Arm X', 'Right Arm Y', 'Right Arm Z',  # 7-9
                'Right Arm Roll', 'Right Arm Pitch', 'Right Arm Yaw',  # 10-12
                'Right Gripper',  # 13
                'Velocity X (model output)', 'Velocity Y (model output)', 'Velocity Yaw (model output)',  # 14-16
                'Chassis Pos X (integrated)', 'Chassis Pos Y (integrated)', 'Chassis Pos Yaw (integrated)',  # 17-19
                'Height (Lift)',  # 20
                'Head Yaw', 'Head Pitch'  # 21-22
            ]
            
            # Collect data for plotting
            # Format: [(current_val, pred_seq), ...]
            
            for i in range(7):
                curr = curr_arm1[i]
                if pred_arm1 is not None and pred_arm1.ndim == 2 and pred_arm1.shape[1] > i:
                    pred = pred_arm1[:, i]
                    x = np.arange(pred.shape[0])  # 0, 1, ..., 800，首点即 agent_pos
                else:
                    pred = np.array([])
                    x = np.array([])

                ax = axes[i]
                ax.plot([0], [curr], "bo", markersize=8, label="current state")
                if pred.size > 0:
                    ax.plot(x, pred, "r.-", label="interpolated trajectory")
                if x_raw_1 is not None and y_raw_1 is not None and y_raw_1.shape[1] > i:
                    ax.plot(x_raw_1, y_raw_1[:, i], "g^", markersize=6, label="raw model output")
                ax.set_title(f"{action_labels[i]}")
                ax.grid(True)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)

            # Right Arm (7-13)
            for i in range(7):
                curr = curr_arm2[i]
                if pred_arm2 is not None and pred_arm2.ndim == 2 and pred_arm2.shape[1] > i:
                    pred = pred_arm2[:, i]
                    x = np.arange(pred.shape[0])
                else:
                    pred = np.array([])
                    x = np.array([])

                idx = i + 7
                ax = axes[idx]
                ax.plot([0], [curr], "bo", markersize=8, label="current state")
                if pred.size > 0:
                    ax.plot(x, pred, "r.-", label="interpolated trajectory")
                if x_raw_2 is not None and y_raw_2 is not None and y_raw_2.shape[1] > i:
                    ax.plot(x_raw_2, y_raw_2[:, i], "g^", markersize=6, label="raw model output")
                ax.set_title(f"{action_labels[idx]}")
                ax.grid(True)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)

            for i in range(3):
                curr = curr_chassis_vel[i]
                if pred_chassis_vel is not None and pred_chassis_vel.ndim == 2 and pred_chassis_vel.shape[1] > i:
                    pred = pred_chassis_vel[:, i]
                    x = np.arange(pred.shape[0])
                else:
                    pred = np.array([])
                    x = np.array([])    

                idx = i + 14
                ax = axes[idx]
                ax.plot([0], [curr], "bo", markersize=8, label="velocity obs fed to model")
                if pred.size > 0:
                    ax.plot(x, pred, "r.-", label="model output velocity")
                ax.set_title(f"{action_labels[idx]}")
                ax.grid(True)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)

            for i in range(3):
                curr = curr_chassis_pos[i]
                if pred_chassis_pos is not None and pred_chassis_pos.ndim == 2 and pred_chassis_pos.shape[1] > i:
                    pred = pred_chassis_pos[:, i]
                    x = np.arange(pred.shape[0])
                else:
                    pred = np.array([])
                    x = np.array([])

                idx = i + 17
                ax = axes[idx]
                ax.plot([0], [curr], "bo", markersize=8, label="current chassis pose")
                if pred.size > 0:
                    ax.plot(x, pred, "r.-", label="velocity-integrated pose (interpolated)")
                ax.set_title(f"{action_labels[idx]}")
                ax.grid(True)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)

            # Lift (20)
            idx = 20
            curr = curr_lift[0] if len(curr_lift) > 0 else 0
            if pred_lift is not None and pred_lift.ndim == 2 and pred_lift.shape[1] >= 1:
                pred = pred_lift[:, 0]
                x = np.arange(pred.shape[0]) + 1
            else:
                pred = np.array([])
                x = np.array([])

            ax = axes[idx]
            ax.plot([0], [curr], "bo", markersize=8, label="agent_pos")
            if pred.size > 0:
                ax.plot(x, pred, "r.-", label="raw_action(action_before_interpolation)")
            ax.set_title(f"{action_labels[idx]}")
            ax.grid(True)

            # Head (21-22)
            for i in range(2):
                curr = curr_head[i]
                if pred_head is not None and pred_head.ndim == 2 and pred_head.shape[1] > i:
                    pred = pred_head[:, i]
                    x = np.arange(pred.shape[0]) + 1
                else:
                    pred = np.array([])
                    x = np.array([])

                idx = i + 21
                ax = axes[idx]
                ax.plot([0], [curr], "bo", markersize=8, label="agent_pos")
                if pred.size > 0:
                    ax.plot(x, pred, "r.-", label="raw_action(action_before_interpolation)")
                ax.set_title(f"{action_labels[idx]}")
                ax.grid(True)

            plt.tight_layout()
            # Atomic save: write tmp then rename. Use .tmp.png so matplotlib recognizes PNG format.
            save_path = os.path.join(self.plot_dir, f"chunk_{self.plot_counter:04d}.png")
            tmp_path = save_path + ".tmp.png"
            plt.savefig(tmp_path)
            os.replace(tmp_path, save_path)
            plt.close(fig)
            self.plot_counter += 1
            logger.info(f"[DEBUG PLOT] Saved chunk plot: {save_path}")

        except Exception as e:
            logger.warning(f"Failed to visualize actions: {e}")
            try:
                plt.close('all')
            except:
                pass    

    def _execute_actions(self, outputs: Dict):
        """Execute actions."""
        # 2. Decode actions
        arm1_actions, arm2_actions = self._decode_arm_actions(outputs)
        (
            head_actions,
            lift_actions,
            chassis_vel_actions,
            car_pose_odom_actions,
        ) = self._decode_auxiliary_actions(outputs)

        # Update last odom actions for next iteration velocity calculation
        self.last_car_pose_odom_actions = car_pose_odom_actions
        # Save raw (pre-interpolation) model output for visualization
        raw_arm1 = np.asarray(arm1_actions).copy() if arm1_actions is not None else None
        raw_arm2 = np.asarray(arm2_actions).copy() if arm2_actions is not None else None
        
        arm1_actions = self._interpolate_arm_actions(arm1_actions)
        arm2_actions = self._interpolate_arm_actions(arm2_actions)
        if arm1_actions is None or len(arm1_actions) == 0:
            return

        # Restore gripper from raw to avoid interpolation artifacts
        if self.interpolate_multiplier > 1 and raw_arm1 is not None:
            arm1_actions = restore_gripper_from_raw(
                arm1_actions, raw_arm1, self.interpolate_multiplier
            )
        if self.interpolate_multiplier > 1 and raw_arm2 is not None:
            arm2_actions = restore_gripper_from_raw(
                arm2_actions, raw_arm2, self.interpolate_multiplier
            )
        steps = len(arm1_actions)
        dt_model = 1.0 / self.camera_capture_hz
        pose_actions = self._prepare_chassis_trajectory(chassis_vel_actions, car_pose_odom_actions, dt_model)
        if self.save_debug_plot:
            self._visualize_actions(arm1_actions, arm2_actions, head_actions, lift_actions, pose_actions,
            chassis_vel_actions=chassis_vel_actions,
            raw_arm1=raw_arm1,
            raw_arm2=raw_arm2,
            )   
        self._execute_control_loop(steps, arm1_actions, arm2_actions, head_actions, lift_actions, pose_actions)
        if pose_actions is not None and len(pose_actions) > 0:
            self._wait_for_chassis_target(pose_actions[-1], timeout=5.0)
