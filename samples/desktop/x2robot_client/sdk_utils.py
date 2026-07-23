import cv2
import numpy as np
from scipy.spatial import transform
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
        mode: "end_pose" (handles Euler angles with NLERP) or "joints"

        The final gripper dimension uses zero-order hold in both modes so grasp and
        release commands remain aligned with the original action keyframes.

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
        # Linear interp for position (0,1,2).
        # Assuming format: [x, y, z, r, p, y, gripper]
        # If dims > 7, we might need to be careful, but assuming 7D for now as per EX001

        # Interpolate non-orientation parts (0,1,2)
        for i in range(3):
            interpolated_actions[:, i] = np.interp(
                target_indices, original_indices, actions_np[:, i]
            )

        # SLERP for orientation (3,4,5) represented as Euler angles.
        if action_dim >= 6:
            rotations = transform.Rotation.from_euler(
                "xyz", actions_np[:, 3:6], degrees=False
            )
            interpolated_actions[:, 3:6] = transform.Slerp(
                original_indices, rotations
            )(target_indices).as_euler("xyz", degrees=False)

    else:
        # The last dimension is the gripper; only interpolate arm joints here.
        for i in range(action_dim - 1):
            interpolated_actions[:, i] = np.interp(
                target_indices, original_indices, actions_np[:, i]
            )

    has_gripper = action_dim > 6 if mode == "end_pose" else action_dim > 0
    if has_gripper:
        # Hold the previous command until the next model keyframe. The actuator's
        # own position controller handles the physical opening/closing trajectory.
        gripper_indices = np.arange(int(target_num_actions)) // factor
        gripper_indices = np.minimum(gripper_indices, num_actions - 1)
        interpolated_actions[:, -1] = actions_np[gripper_indices, -1]

    return interpolated_actions.tolist()


def stitch_trajectory(
    current_action: list,
    predicted_actions: list,
    elapsed_steps: int,
    blend_steps: int,
    mode: str = "end_pose",
) -> list:
    """Drop stale predictions and blend the arm into the remaining trajectory."""
    if not predicted_actions:
        return []

    skip = max(0, int(elapsed_steps))
    if skip >= len(predicted_actions):
        return []

    remaining = [list(action) for action in predicted_actions[skip:]]
    if current_action is None or blend_steps <= 0:
        return remaining

    transition = interpolate_trajectory(
        [list(current_action), remaining[0]], max(1, int(blend_steps)), mode
    )
    return transition[1:] + remaining[1:]


def crossfade_trajectories(
    existing_actions: list,
    predicted_actions: list,
    blend_steps: int,
    mode: str = "end_pose",
) -> list:
    """Blend an existing trajectory into a replacement without a velocity step."""
    if not predicted_actions:
        return []
    if not existing_actions or blend_steps <= 0:
        return [list(action) for action in predicted_actions]

    overlap = min(len(existing_actions), len(predicted_actions), int(blend_steps))
    blended = []
    for index in range(overlap):
        old = np.asarray(existing_actions[index], dtype=float)
        new = np.asarray(predicted_actions[index], dtype=float)
        progress = (index + 1) / overlap
        weight = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
        action = (1.0 - weight) * old + weight * new

        if mode == "end_pose" and len(action) >= 6:
            old_quat = transform.Rotation.from_euler("xyz", old[3:6]).as_quat()
            new_quat = transform.Rotation.from_euler("xyz", new[3:6]).as_quat()
            if np.dot(old_quat, new_quat) < 0:
                new_quat = -new_quat
            quat = (1.0 - weight) * old_quat + weight * new_quat
            quat /= max(np.linalg.norm(quat), np.finfo(float).eps)
            action[3:6] = transform.Rotation.from_quat(quat).as_euler("xyz")

        # Gripper remains event-based during the overlap.
        action[-1] = new[-1] if index == overlap - 1 else old[-1]
        blended.append(action.tolist())

    blended.extend([list(action) for action in predicted_actions[overlap:]])
    return blended


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
