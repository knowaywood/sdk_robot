import cv2
import numpy as np
from scipy.spatial import transform
from scipy.spatial.transform import Slerp
from x2robot.sensor_msgs import CompressedImage


def compressed_image_to_numpy(compressed_image_msg):
    """
    将 sensor_msgs.msg.CompressedImage 转换为 (h, w, c) 的 RGB 格式 NumPy 数组。

    参数:
        compressed_image_msg (sensor_msgs.msg.CompressedImage): ROS 的压缩图像消息。

    返回:
        numpy.ndarray: (h, w, c) 的 RGB 格式 NumPy 数组。
    """
    # 获取压缩图像数据
    compressed_data = compressed_image_msg.data

    # 将字节数据解码为图像
    np_arr = np.frombuffer(compressed_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 解码为 BGR 格式

    # 将 BGR 转换为 RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return rgb_image


def compose(f, g):
    """返回一个嵌套函数对象 h(x) = f(g(x))"""

    def h():
        return f(g())

    return h


def interpolate_trajectory(actions: list, factor: int, mode: str = "end_pose") -> list:
    """
    Interpolate trajectory to increase density of points.

    Args:
        actions: List of action points
        factor: Interpolation multiplier (e.g. 2 means 2x points)
        mode: "end_pose" (handles Euler angles with Slerp) or "joints" (all linear)

    Returns:
        Interpolated list of actions
    """
    if not actions or len(actions) < 2 or factor <= 1:
        return actions

    actions_np = np.array(actions)
    num_actions, action_dim = actions_np.shape

    target_num_actions = (num_actions - 1) * factor + 1

    original_indices = np.linspace(0, num_actions - 1, num_actions)
    target_indices = np.linspace(0, num_actions - 1, int(target_num_actions))

    interpolated_actions = np.zeros((int(target_num_actions), action_dim))

    if mode == "end_pose":
        # Linear interp for pos (0,1,2) and gripper (6)
        # Assuming format: [x, y, z, r, p, y, gripper]
        # If dims > 7, we might need to be careful, but assuming 7D for now as per EX001

        # Interpolate non-orientation parts (0,1,2)
        for i in range(3):
            interpolated_actions[:, i] = np.interp(
                target_indices, original_indices, actions_np[:, i]
            )

        # Interpolate gripper (last element)
        if action_dim > 6:
            interpolated_actions[:, -1] = np.interp(
                target_indices, original_indices, actions_np[:, -1]
            )

        # NLERP for orientation (3,4,5) - Euler angles
        if action_dim >= 6:
            # Convert to quat, interpolate, convert back
            quaternions = transform.Rotation.from_euler(
                "xyz", actions_np[:, 3:6], degrees=False
            ).as_quat()
            interpolated_quats = np.zeros((int(target_num_actions), 4))

            for i in range(4):
                interpolated_quats[:, i] = np.interp(
                    target_indices, original_indices, quaternions[:, i]
                )

            # Normalize quaternions
            norms = np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid div by zero
            interpolated_quats /= norms

            interpolated_actions[:, 3:6] = transform.Rotation.from_quat(
                interpolated_quats
            ).as_euler("xyz", degrees=False)

    else:
        # Linear interp for all joints + gripper
        for i in range(action_dim):
            interpolated_actions[:, i] = np.interp(
                target_indices, original_indices, actions_np[:, i]
            )

    return interpolated_actions.tolist()


def _validate_trajectory(actions: list) -> np.ndarray:
    """Convert an action trajectory to a finite 2-D float array."""
    actions_np = np.asarray(actions, dtype=np.float64)
    if actions_np.ndim != 2 or actions_np.shape[0] == 0:
        raise ValueError("actions must be a non-empty 2-D trajectory")
    if not np.isfinite(actions_np).all():
        raise ValueError("actions contain NaN or infinity")
    return actions_np


def _zero_order_hold(values: np.ndarray, source_times, target_times) -> np.ndarray:
    """Resample event-like values without creating intermediate commands."""
    indices = np.searchsorted(source_times, target_times, side="right") - 1
    indices = np.clip(indices, 0, len(source_times) - 1)
    return values[indices]


def resample_trajectory(
    actions: list,
    source_hz: float,
    target_hz: float,
    mode: str = "end_pose",
) -> list:
    """Resample a model trajectory onto the robot execution time grid.

    Position and joint channels are linearly interpolated. End-effector
    orientation is interpolated on SO(3) with Slerp. The last channel is
    treated as the gripper command and uses zero-order hold so resampling does
    not create unintended intermediate opening/closing commands.
    """
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("source_hz and target_hz must be positive")

    actions_np = _validate_trajectory(actions)
    num_actions, action_dim = actions_np.shape
    if num_actions == 1:
        return actions_np.tolist()

    source_times = np.arange(num_actions, dtype=np.float64) / float(source_hz)
    duration = source_times[-1]
    target_count = max(2, int(round(duration * float(target_hz))) + 1)
    target_times = np.linspace(0.0, duration, target_count, dtype=np.float64)
    output = np.empty((target_count, action_dim), dtype=np.float64)

    if mode == "end_pose":
        if action_dim < 7:
            raise ValueError(
                "end_pose actions must contain [x, y, z, roll, pitch, yaw, gripper]"
            )

        for dim in range(3):
            output[:, dim] = np.interp(
                target_times, source_times, actions_np[:, dim]
            )

        rotations = transform.Rotation.from_euler("xyz", actions_np[:, 3:6])
        output[:, 3:6] = Slerp(source_times, rotations)(target_times).as_euler(
            "xyz"
        )

        # Preserve any extra continuous channels between orientation and grip.
        for dim in range(6, action_dim - 1):
            output[:, dim] = np.interp(
                target_times, source_times, actions_np[:, dim]
            )
    else:
        for dim in range(action_dim - 1):
            output[:, dim] = np.interp(
                target_times, source_times, actions_np[:, dim]
            )

    output[:, -1] = _zero_order_hold(
        actions_np[:, -1], source_times, target_times
    )
    return output.tolist()


def blend_action_pair(old_action, new_action, alpha: float, mode: str):
    """Blend motion channels while keeping the old gripper command.

    The caller switches to the new gripper command only when the overlap is
    complete. This prevents chunk stitching from inventing intermediate grip
    values.
    """
    old = np.asarray(old_action, dtype=np.float64)
    new = np.asarray(new_action, dtype=np.float64)
    if old.shape != new.shape or old.ndim != 1:
        raise ValueError("old_action and new_action must be equal-size vectors")
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("actions contain NaN or infinity")

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = old.copy()
    if mode == "end_pose":
        if old.size < 7:
            raise ValueError(
                "end_pose actions must contain [x, y, z, roll, pitch, yaw, gripper]"
            )
        blended[:3] = (1.0 - alpha) * old[:3] + alpha * new[:3]
        rotations = transform.Rotation.from_euler(
            "xyz", np.stack([old[3:6], new[3:6]])
        )
        blended[3:6] = Slerp([0.0, 1.0], rotations)([alpha]).as_euler("xyz")[0]
        if old.size > 7:
            blended[6:-1] = (1.0 - alpha) * old[6:-1] + alpha * new[6:-1]
    else:
        blended[:-1] = (1.0 - alpha) * old[:-1] + alpha * new[:-1]

    # Do not blend the gripper. Change it only at the end of the overlap.
    blended[-1] = new[-1] if alpha >= 1.0 else old[-1]
    return blended.tolist()


def smoothen(
    arr: np.ndarray, window: int = None, poly: int = 2, ema_alpha: float = 0.25
) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    T, D = a.shape
    if T < 3:
        return a.copy()
    if window is None:
        w = int(max(3, round(T / 5)))
    else:
        w = int(max(3, window))
    if w % 2 == 0:
        w += 1
    if w >= T:
        w = T - 1 if (T - 1) % 2 == 1 else T - 2
    if w < 3:
        w = 3
    p = int(max(1, min(poly, w - 1)))
    sm = np.empty_like(a)
    try:
        from scipy.signal import savgol_filter

        for d in range(D):
            sm[:, d] = savgol_filter(
                a[:, d], window_length=w, polyorder=p, mode="interp"
            )
    except Exception:
        alpha = float(ema_alpha)

        def ema_two_sided(x: np.ndarray) -> np.ndarray:
            y = np.empty_like(x)
            y[0] = x[0]
            for i in range(1, T):
                y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
            z = np.empty_like(x)
            z[-1] = x[-1]
            for i in range(T - 2, -1, -1):
                z[i] = alpha * x[i] + (1.0 - alpha) * z[i + 1]
            return 0.5 * (y + z)

        for d in range(D):
            sm[:, d] = ema_two_sided(a[:, d])
    return sm.astype(np.float32)


def compressed_image_to_rgb(compressed_images):
    """Converts a list of ROS CompressedImage messages to a list of RGB NumPy arrays."""
    if not isinstance(compressed_images, list):
        compressed_images = [compressed_images]

    rgb_images = []
    for compressed_image in compressed_images:
        if not isinstance(compressed_image, CompressedImage):
            raise TypeError(
                f"Expected sensor_msgs.msg.CompressedImage, but got {type(compressed_image)}"
            )

        # Defensively check if the image data is valid
        if not compressed_image.data:
            raise ValueError("CompressedImage data is empty")

        # try:
        # The image data is a bytes-like object from the message
        np_arr = np.frombuffer(compressed_image.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            rgb_images.append(None)
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb_images.append(img_rgb)
        # except Exception:
        #     # If any other error occurs during conversion, append None
        #     rgb_images.append(None)

    return rgb_images


def interpolate_arm_trajectory(traj, original_indices, target_indices, target_length):
    """优化的机械臂轨迹插值"""
    if traj.ndim == 1:
        traj = traj.reshape(-1, 1)
    dims = traj.shape[1]
    interpolated = np.zeros((target_length, dims))
    for i in range(dims):
        interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])
    return interpolated


def interpolate_position_trajectory(
    traj, original_indices, target_indices, target_length
):
    """优化的位置轨迹插值"""
    if traj.ndim == 1:
        traj = traj.reshape(-1, 1)
    dims = traj.shape[1]
    interpolated = np.zeros((target_length, dims))
    for i in range(dims):
        interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])
    return interpolated


def interpolate_head_trajectory(traj, original_indices, target_indices, target_length):
    """优化的头部轨迹插值"""
    if traj.ndim == 1:
        traj = traj.reshape(-1, 1)
    dims = traj.shape[1]
    interpolated = np.zeros((target_length, dims))
    for i in range(dims):
        interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])
    return interpolated


def chassis_T2pose(T: np.ndarray) -> list:
    """
    @param T : 4x4 transformation matrix
    @return: list of length 3, [x,y,yaw]
    """
    if T.shape != (4, 4):
        raise ValueError("T must be a 4x4 matrix.")
    pos = T[0:3, 3]
    euler = transform.Rotation.from_matrix(T[0:3, 0:3]).as_euler("xyz")
    return [pos[0], pos[1], euler[2]]


def chassis_pose2T(pose: list) -> np.ndarray:
    """
    @param pose: list of length 3, [x,y,yaw]
    @return: 4x4 transformation matrix
    """
    if len(pose) != 3:
        raise ValueError("pose must be a list of length 3.")
    pos = np.array(pose[0:3])
    yaw = pose[2]
    rot = transform.Rotation.from_euler("xyz", [0, 0, yaw])
    T = np.eye(4)
    T[0:3, 0:3], T[0:3, 3] = rot.as_matrix(), pos
    return T


def qua2euler(quaternion: list) -> list:
    """
    Converts a quaternion to Euler angles.
    @param quaternion: list of length 4, [x, y, z, w]
    @return: list of length 3, [roll, pitch, yaw]
    """
    if len(quaternion) != 4:
        raise ValueError("quaternion must be a list of length 4.")
    return transform.Rotation.from_quat(quaternion).as_euler("xyz").tolist()


def euler2qua(euler: list) -> list:
    """
    Converts Euler angles to a quaternion.
    @param euler: list of length 3, [roll, pitch, yaw] in 'xyz' order.
    @return: list of length 4, [x, y, z, w]
    """
    if len(euler) != 3:
        raise ValueError("Euler angles must be a list of length 3.")
    # The from_euler method creates a Rotation object.
    # The as_quat method returns the quaternion [x, y, z, w].
    quat = transform.Rotation.from_euler("xyz", euler).as_quat()
    return quat.tolist()
