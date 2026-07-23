import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2robot_client.sdk_utils import (
    crossfade_trajectories,
    interpolate_trajectory,
    stitch_trajectory,
)


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

    def test_end_pose_orientation_takes_short_path_across_pi_boundary(self):
        actions = [
            [0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(179.0), 4.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-179.0), 4.5],
        ]

        result = np.asarray(interpolate_trajectory(actions, factor=10))
        rotations = transform.Rotation.from_euler("xyz", result[:, 3:6])
        step_angles = (rotations[:-1].inv() * rotations[1:]).magnitude()

        self.assertLess(np.rad2deg(step_angles.max()), 0.21)

    def test_end_pose_position_has_continuous_velocity_at_keyframe(self):
        actions = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5],
            [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5],
        ]

        result = np.asarray(interpolate_trajectory(actions, factor=10))
        step_before = result[10, 0] - result[9, 0]
        step_after = result[11, 0] - result[10, 0]

        self.assertLess(abs(step_before - step_after), 0.01)
        self.assertGreaterEqual(result[:, 0].min(), 0.0)
        self.assertLessEqual(result[:, 0].max(), 1.2)

    def test_end_pose_rotation_has_continuous_rate_at_keyframe(self):
        actions = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 4.5],
        ]

        result = np.asarray(interpolate_trajectory(actions, factor=10))
        rotations = transform.Rotation.from_euler("xyz", result[:, 3:6])
        step_angles = (rotations[:-1].inv() * rotations[1:]).magnitude()

        self.assertLess(abs(step_angles[9] - step_angles[10]), 0.02)


class StitchTrajectoryTest(unittest.TestCase):
    def test_drops_elapsed_actions_and_blends_from_current_command(self):
        current = [-1.0, 4.5]
        predicted = [
            [0.0, 4.5],
            [1.0, 4.5],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]

        result = np.asarray(
            stitch_trajectory(
                current,
                predicted,
                elapsed_steps=2,
                blend_steps=2,
                mode="joints",
            )
        )

        np.testing.assert_allclose(result[:, 0], [0.5, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result[:, -1], [4.5, 0.0, 0.0, 0.0])

    def test_rejects_prediction_when_entire_chunk_is_stale(self):
        result = stitch_trajectory(
            [0.0, 4.5],
            [[1.0, 4.5], [2.0, 0.0]],
            elapsed_steps=2,
            blend_steps=2,
            mode="joints",
        )

        self.assertEqual(result, [])


class CrossfadeTrajectoriesTest(unittest.TestCase):
    def test_uses_minimum_jerk_weights_and_prioritizes_gripper_close(self):
        existing = [[0.0, 4.5], [1.0, 4.5], [2.0, 4.5], [3.0, 4.5]]
        predicted = [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]]

        result = np.asarray(
            crossfade_trajectories(existing, predicted, 4, mode="joints")
        )

        np.testing.assert_allclose(
            result[:, 0], [0.0, 1.5, 3.79296875, 6.0]
        )
        np.testing.assert_allclose(result[:, -1], [0.0, 0.0, 0.0, 0.0])

    def test_delays_gripper_open_until_arm_handoff_finishes(self):
        existing = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        predicted = [[0.0, 4.5], [2.0, 4.5], [4.0, 4.5], [6.0, 4.5]]

        result = np.asarray(
            crossfade_trajectories(existing, predicted, 4, mode="joints")
        )

        np.testing.assert_allclose(result[:, -1], [0.0, 0.0, 0.0, 4.5])


if __name__ == "__main__":
    unittest.main()
