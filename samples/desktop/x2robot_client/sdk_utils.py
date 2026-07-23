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


def restore_gripper_from_raw(
    interpolated: list, raw: list, factor: int
) -> list:
    """Override the last dim (gripper) of interpolated arm actions with
    nearest-neighbour values from the raw (pre-interpolation) actions,
    preventing linear interpolation artifacts on the binary gripper."""
    if not interpolated or not raw or factor <= 1:
        return interpolated
    raw_arr = np.asarray(raw, dtype=np.float64)
    out = np.asarray(interpolated, dtype=np.float64)
    if raw_arr.ndim != 2 or out.ndim != 2 or raw_arr.shape[1] != out.shape[1]:
        return interpolated
    raw_len = raw_arr.shape[0]
    idx = np.arange(out.shape[0]) // factor
    np.clip(idx, 0, raw_len - 1, out=idx)
    out[:, -1] = raw_arr[idx, -1]
    return out.tolist()


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
