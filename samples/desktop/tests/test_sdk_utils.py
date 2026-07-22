import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2robot_client.sdk_utils import interpolate_trajectory


class InterpolateTrajectoryTest(unittest.TestCase):
    def test_end_pose_holds_gripper_until_next_keyframe(self):
        actions = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        ]

        result = np.asarray(interpolate_trajectory(actions, factor=4))

        np.testing.assert_allclose(result[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(result[:, -1], [4.5, 4.5, 4.5, 4.5, 0.0])

    def test_joints_interpolates_joints_but_holds_gripper(self):
        actions = [
            [0.0, 0.0, 4.5],
            [2.0, 4.0, 0.0],
        ]

        result = np.asarray(
            interpolate_trajectory(actions, factor=2, mode="joints")
        )

        np.testing.assert_allclose(
            result[:, :-1], [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]
        )
        np.testing.assert_allclose(result[:, -1], [4.5, 4.5, 0.0])

    def test_gripper_switches_at_each_original_keyframe(self):
        actions = [
            [0.0, 4.5],
            [1.0, 0.0],
            [2.0, 3.0],
        ]

        result = np.asarray(
            interpolate_trajectory(actions, factor=3, mode="joints")
        )

        np.testing.assert_allclose(result[:, 0], np.linspace(0.0, 2.0, 7))
        np.testing.assert_allclose(
            result[:, -1], [4.5, 4.5, 4.5, 0.0, 0.0, 0.0, 3.0]
        )


if __name__ == "__main__":
    unittest.main()
