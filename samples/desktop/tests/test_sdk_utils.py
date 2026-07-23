import sys
import types
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import transform


try:
    import x2robot.sensor_msgs  # noqa: F401
except ImportError:
    x2robot_module = types.ModuleType("x2robot")
    sensor_msgs_module = types.ModuleType("x2robot.sensor_msgs")
    sensor_msgs_module.CompressedImage = type("CompressedImage", (), {})
    x2robot_module.sensor_msgs = sensor_msgs_module
    sys.modules.setdefault("x2robot", x2robot_module)
    sys.modules["x2robot.sensor_msgs"] = sensor_msgs_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2robot_client.sdk_utils import blend_action_pair, resample_trajectory


class ResampleTrajectoryTest(unittest.TestCase):
    def test_30_hz_chunk_is_resampled_to_50_hz_by_time(self):
        actions = np.zeros((32, 7), dtype=np.float64)
        actions[:, 0] = np.arange(32) / 30.0
        actions[16:, -1] = 4.5

        result = np.asarray(
            resample_trajectory(actions.tolist(), 30.0, 50.0, "end_pose")
        )

        self.assertEqual(result.shape, (53, 7))
        np.testing.assert_allclose(result[0], actions[0], atol=1e-7)
        np.testing.assert_allclose(result[-1], actions[-1], atol=1e-7)
        self.assertTrue(set(np.unique(result[:, -1])).issubset({0.0, 4.5}))

    def test_chunk_blend_does_not_interpolate_gripper(self):
        old = [0.0, 0.0, 0.0, 0.0, 0.0, 3.12, 0.0]
        new = [1.0, 0.0, 0.0, 0.0, 0.0, -3.12, 4.5]

        midpoint = blend_action_pair(old, new, 0.5, "end_pose")
        endpoint = blend_action_pair(old, new, 1.0, "end_pose")

        self.assertEqual(midpoint[-1], old[-1])
        self.assertEqual(endpoint[-1], new[-1])
        old_rotation = transform.Rotation.from_euler("xyz", old[3:6])
        midpoint_rotation = transform.Rotation.from_euler("xyz", midpoint[3:6])
        angular_distance = (old_rotation.inv() * midpoint_rotation).magnitude()
        self.assertLess(angular_distance, 0.03)


if __name__ == "__main__":
    unittest.main()
