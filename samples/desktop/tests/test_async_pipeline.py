import sys
import unittest
from queue import Queue
from pathlib import Path

import numpy as np
from scipy.spatial import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2robot_client.async_desktop_sdk_client import (
    AsyncDesktopClient,
    _InferenceRequest,
    _PredictionChunk,
    _next_control_deadline,
)


class _RecordingGripper:
    def __init__(self):
        self.positions = []

    def set_position(self, command):
        self.positions.append(command.position)


def _make_client():
    client = AsyncDesktopClient.__new__(AsyncDesktopClient)
    client.control_mode = "joints"
    client.interpolate_multiplier = 2
    client.prefetch_margin = 0.1
    client.blend_steps = 2
    client.world_lock_steps = 3
    client.replan_steps = 10
    client.control_period = 0.1
    client.max_linear_speed = 0.2
    client.max_angular_speed = 1.0
    client._motion_limit_count = 0
    client._inference_clients = [object(), object()]
    client._inference_in_flight = set()
    client._request_queue = Queue(maxsize=2)
    client._prediction_queue = Queue(maxsize=4)
    client._request_counter = 0
    client._discard_before_request_id = 0
    client._last_applied_request_id = 0
    client.gripper_deadband = 0.05
    client.gripper_close_confirm = 0.15
    client.gripper_release_confirm = 0.4
    client.gripper_reopen_dwell = 1.0
    client.gripper_release_travel = 0.05
    client.gripper_approach_travel = 0.03
    client.gripper_transition_linear_speed = 0.08
    client.gripper_transition_angular_speed = 0.6
    client.GRIPPER_DATA_MAX = 4.5
    client._last_gripper_command = {"left": None, "right": None}
    client._last_gripper_sent_at = {"left": 0.0, "right": 0.0}
    client._gripper_transition_candidate = {"left": None, "right": None}
    client._gripper_closed_pose = {"left": None, "right": None}
    client._gripper_open_pose = {"left": None, "right": None}
    client._gripper_block_counts = {"motion": 0, "approach": 0, "travel": 0}
    return client


class AsyncPipelineTest(unittest.TestCase):
    def test_control_deadline_never_catches_up_with_a_burst(self):
        next_tick, missed = _next_control_deadline(1.0, 1.015, 0.01)
        self.assertTrue(missed)
        self.assertAlmostEqual(next_tick, 1.025)

        next_tick, missed = _next_control_deadline(2.0, 2.004, 0.01)
        self.assertFalse(missed)
        self.assertAlmostEqual(next_tick, 2.01)

    def test_replan_waits_for_committed_action_window(self):
        client = _make_client()

        self.assertTrue(client._replan_due(False, 10, 11))
        self.assertFalse(client._replan_due(True, 10, 19))
        self.assertTrue(client._replan_due(True, 10, 20))

    def test_prediction_drops_elapsed_prefix_and_keeps_future_stage(self):
        client = _make_client()
        request = _InferenceRequest(request_id=1, control_step=10)
        prediction = _PredictionChunk(
            request=request,
            outputs={
                "follow1_pos": [[1.0, 4.5], [2.0, 0.0]],
                "follow2_pos": [[1.0, 4.5], [2.0, 0.0]],
            },
            left_anchor=[0.0, 4.5],
            right_anchor=[0.0, 4.5],
            started_at=1.0,
            finished_at=1.2,
        )

        result = client._stitch_prediction(
            prediction,
            control_step=12,
            last_command=([-1.0, 4.5], [-1.0, 4.5]),
            active_actions=[],
        )

        self.assertEqual(
            [pair[0][0] for pair in result], [-1.0, 0.5, 2.0]
        )
        self.assertEqual(
            [pair[0][-1] for pair in result],
            [4.5, 4.5, 0.0],
        )

    def test_long_inference_keeps_enough_actions_for_next_prediction(self):
        client = _make_client()
        prediction = _PredictionChunk(
            request=_InferenceRequest(request_id=1, control_step=10),
            outputs={
                "follow1_pos": [[1.0, 4.5], [2.0, 0.0]],
                "follow2_pos": [[1.0, 4.5], [2.0, 0.0]],
            },
            left_anchor=[0.0, 4.5],
            right_anchor=[0.0, 4.5],
            started_at=1.0,
            finished_at=2.0,
        )

        result = client._stitch_prediction(
            prediction,
            control_step=20,
            last_command=([0.0, 4.5], [0.0, 4.5]),
            active_actions=[],
        )

        self.assertEqual(len(result), 5)
        self.assertEqual([pair[0][-1] for pair in result][-1], 0.0)

    def test_latency_trim_reserves_buffer_based_on_worker_throughput(self):
        client = _make_client()

        self.assertEqual(
            client._latency_trim_steps(
                action_count=20,
                inference_steps=10,
                inference_latency=0.4,
            ),
            10,
        )

        client._inference_clients = [object()]
        self.assertEqual(
            client._latency_trim_steps(
                action_count=20,
                inference_steps=10,
                inference_latency=1.2,
            ),
            7,
        )

    def test_world_locked_joint_handoff_converges_to_absolute_target(self):
        client = _make_client()

        result = client._align_actions_to_world_reference(
            [[2.0, 4.5], [4.0, 4.5], [6.0, 0.0]],
            reference=[0.0, 4.5],
        )

        self.assertEqual([action[0] for action in result], [0.0, 3.0, 6.0])
        self.assertEqual([action[-1] for action in result], [4.5, 4.5, 0.0])

    def test_short_blend_source_is_extended_with_decaying_velocity(self):
        client = _make_client()

        result = client._extend_blend_trajectory(
            [[0.0, 4.5], [1.0, 4.5]],
            previous_action=[-1.0, 4.5],
            target_steps=5,
        )

        np.testing.assert_allclose(
            [action[0] for action in result], [0.0, 1.0, 1.75, 2.25, 2.5]
        )
        self.assertEqual([action[-1] for action in result], [4.5] * 5)

    def test_end_pose_speed_limiter_caps_translation_and_rotation(self):
        client = _make_client()
        client.control_mode = "end_pose"
        previous = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5]
        target = [0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 4.5]

        result = client._limit_end_pose_action(target, previous)
        result_rotation = transform.Rotation.from_euler("xyz", result[3:6])

        self.assertAlmostEqual(result[0], 0.02)
        self.assertAlmostEqual(result_rotation.magnitude(), 0.1)
        self.assertEqual(result[-1], 4.5)
        self.assertEqual(client._motion_limit_count, 1)

    def test_world_locked_end_pose_handoff_keeps_absolute_target(self):
        client = _make_client()
        client.control_mode = "end_pose"
        actions = [
            [1.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-179.0), 4.5],
            [2.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-178.0), 4.5],
            [3.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-177.0), 0.0],
        ]
        reference = [0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(179.0), 4.5]

        result = np.asarray(
            client._align_actions_to_world_reference(actions, reference)
        )
        rotations = transform.Rotation.from_euler("xyz", result[:, 3:6])
        reference_rotation = transform.Rotation.from_euler("xyz", reference[3:6])
        target_rotation = transform.Rotation.from_euler("xyz", actions[-1][3:6])

        self.assertAlmostEqual(result[0, 0], reference[0])
        self.assertAlmostEqual(result[-1, 0], actions[-1][0])
        self.assertLess((reference_rotation.inv() * rotations[0]).magnitude(), 1e-8)
        self.assertLess((target_rotation.inv() * rotations[-1]).magnitude(), 1e-8)
        self.assertEqual(result[:, -1].tolist(), [4.5, 4.5, 0.0])

    def test_request_queue_allows_one_request_per_worker(self):
        client = _make_client()

        self.assertTrue(client._request_prediction(control_step=0))
        self.assertTrue(client._request_prediction(control_step=1))
        self.assertFalse(client._request_prediction(control_step=2))
        self.assertEqual(client._inference_in_flight, {1, 2})

    def test_stale_worker_error_does_not_interrupt_newer_prediction(self):
        client = _make_client()
        client._last_applied_request_id = 2
        client._inference_in_flight.add(1)
        client._prediction_queue.put_nowait(
            _PredictionChunk(
                request=_InferenceRequest(request_id=1, control_step=0),
                outputs=None,
                left_anchor=None,
                right_anchor=None,
                started_at=1.0,
                finished_at=1.2,
                error=RuntimeError("stale failure"),
            )
        )

        self.assertIsNone(client._poll_prediction())
        self.assertNotIn(1, client._inference_in_flight)

    def test_gripper_observation_is_scaled_to_model_range(self):
        client = _make_client()
        client.GRIPPER_DATA_MAX = 1.55
        inputs = {
            "state": {
                "follow1_pos": np.array([1.0, 4.5], dtype=np.float32),
                "follow2_pos": np.array([2.0, 2.25], dtype=np.float32),
                "follow1_gripper": np.array(4.5, dtype=np.float32),
                "follow2_gripper": np.array(2.25, dtype=np.float32),
            }
        }

        client._normalize_gripper_observation(inputs)

        self.assertAlmostEqual(inputs["state"]["follow1_pos"][-1], 1.55)
        self.assertAlmostEqual(inputs["state"]["follow2_pos"][-1], 0.775)
        self.assertAlmostEqual(inputs["state"]["follow1_gripper"], 1.55)
        self.assertAlmostEqual(inputs["state"]["follow2_gripper"], 0.775)

    def test_unloaded_gripper_is_preopened_before_planned_approach(self):
        client = _make_client()
        client.GRIPPER_DATA_MAX = 1.55
        client._last_gripper_command["right"] = 0.0
        actions = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.55],
            [3.0, 0.0],
        ]

        preopened = client._preopen_unloaded_gripper("right", actions)

        self.assertTrue(preopened)
        self.assertEqual(
            [action[-1] for action in actions], [1.55, 1.55, 1.55, 0.0]
        )

    def test_grasped_gripper_is_not_preopened(self):
        client = _make_client()
        client._last_gripper_command["right"] = 0.0
        client._gripper_closed_pose["right"] = np.zeros(3)
        actions = [[0.0, 0.0], [1.0, 4.5]]

        preopened = client._preopen_unloaded_gripper("right", actions)

        self.assertFalse(preopened)
        self.assertEqual([action[-1] for action in actions], [0.0, 4.5])

    def test_plan_summary_reports_next_gripper_event(self):
        client = _make_client()
        client.control_mode = "end_pose"
        client._last_gripper_command = {"left": 0.0, "right": 4.5}
        prepared = [
            (
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5],
            ),
            (
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5],
                [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ]

        summary = client._format_plan_summary(prepared)

        self.assertIn("left:0.100/open@0.10s", summary)
        self.assertIn("right:0.200/close@0.10s", summary)

    def test_gripper_release_requires_stable_request_and_dwell(self):
        client = _make_client()
        gripper = _RecordingGripper()
        client._last_gripper_command["left"] = 0.0
        client._last_gripper_sent_at["left"] = 1.0
        client._gripper_closed_pose["left"] = np.zeros(3)

        client._send_gripper_if_needed("left", gripper, 4.0, now=1.2)
        client._send_gripper_if_needed("left", gripper, 4.0, now=1.45)
        client._send_gripper_if_needed("left", gripper, 4.0, now=1.65)
        client._send_gripper_if_needed("left", gripper, 4.0, now=2.1)
        client._send_gripper_if_needed("left", gripper, 4.0, now=3.2)

        self.assertEqual(gripper.positions, [4.5, 4.5])

    def test_unloaded_preopen_is_not_blocked_by_arm_motion_or_release_dwell(self):
        client = _make_client()
        client.control_mode = "end_pose"
        gripper = _RecordingGripper()
        client._last_gripper_command["right"] = 0.0
        client._last_gripper_sent_at["right"] = 1.0
        moving = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5]
        previous = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5]

        client._send_gripper_if_needed(
            "right", gripper, 4.5, now=1.05,
            arm_action=moving, previous_arm_action=previous,
        )
        client._send_gripper_if_needed(
            "right", gripper, 4.5, now=1.25,
            arm_action=moving, previous_arm_action=previous,
        )

        self.assertEqual(gripper.positions, [4.5])
        self.assertEqual(client._gripper_block_counts["motion"], 0)

    def test_gripper_close_waits_for_stable_request_and_low_arm_speed(self):
        client = _make_client()
        client.control_mode = "end_pose"
        gripper = _RecordingGripper()
        client._last_gripper_command["right"] = 4.5
        client._last_gripper_sent_at["right"] = 1.0
        moving = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        previous = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5]
        settled = [0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        client._send_gripper_if_needed(
            "right", gripper, 0.5, now=1.1,
            arm_action=moving, previous_arm_action=previous,
        )
        client._send_gripper_if_needed(
            "right", gripper, 0.5, now=1.3,
            arm_action=moving, previous_arm_action=previous,
        )
        client._send_gripper_if_needed(
            "right", gripper, 0.5, now=1.4,
            arm_action=settled, previous_arm_action=settled,
        )

        self.assertEqual(gripper.positions, [0.0])

    def test_gripper_close_requires_approach_travel_after_opening(self):
        client = _make_client()
        client.control_mode = "end_pose"
        gripper = _RecordingGripper()
        client._last_gripper_command["right"] = 4.5
        client._last_gripper_sent_at["right"] = 1.0
        client._gripper_open_pose["right"] = np.zeros(3)
        too_near = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        approached = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        client._send_gripper_if_needed(
            "right", gripper, 0.0, now=1.0,
            arm_action=too_near, previous_arm_action=too_near,
        )
        client._send_gripper_if_needed(
            "right", gripper, 0.0, now=1.2,
            arm_action=too_near, previous_arm_action=too_near,
        )
        client._send_gripper_if_needed(
            "right", gripper, 0.0, now=1.3,
            arm_action=approached, previous_arm_action=approached,
        )

        self.assertEqual(gripper.positions, [0.0])
        self.assertEqual(client._gripper_block_counts["approach"], 1)

    def test_gripper_release_noise_does_not_open_gripper(self):
        client = _make_client()
        gripper = _RecordingGripper()
        client._last_gripper_command["left"] = 0.0
        client._last_gripper_sent_at["left"] = 1.0

        client._send_gripper_if_needed("left", gripper, 4.0, now=1.05)
        client._send_gripper_if_needed("left", gripper, 2.0, now=1.1)
        client._send_gripper_if_needed("left", gripper, 0.5, now=1.2)

        self.assertEqual(gripper.positions, [])

    def test_gripper_release_requires_travel_after_grasp(self):
        client = _make_client()
        client.control_mode = "end_pose"
        gripper = _RecordingGripper()
        client._last_gripper_command["right"] = 0.0
        client._last_gripper_sent_at["right"] = 0.0
        client._gripper_closed_pose["right"] = np.zeros(3)
        at_grasp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        at_release = [0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        client._send_gripper_if_needed(
            "right", gripper, 4.0, now=1.0,
            arm_action=at_grasp, previous_arm_action=at_grasp,
        )
        client._send_gripper_if_needed(
            "right", gripper, 4.0, now=1.5,
            arm_action=at_grasp, previous_arm_action=at_grasp,
        )
        client._send_gripper_if_needed(
            "right", gripper, 4.0, now=1.6,
            arm_action=at_release, previous_arm_action=at_grasp,
        )
        client._send_gripper_if_needed(
            "right", gripper, 4.0, now=1.7,
            arm_action=at_release, previous_arm_action=at_release,
        )

        self.assertEqual(gripper.positions, [4.5])

    def test_gripper_heartbeat_does_not_reset_grasp_pose(self):
        client = _make_client()
        client.control_mode = "end_pose"
        gripper = _RecordingGripper()
        client._last_gripper_command["right"] = 0.0
        client._last_gripper_sent_at["right"] = 0.0
        client._gripper_closed_pose["right"] = np.zeros(3)
        moved = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        client._send_gripper_if_needed(
            "right",
            gripper,
            0.0,
            now=2.0,
            arm_action=moved,
            previous_arm_action=moved,
        )

        self.assertEqual(gripper.positions, [0.0])
        np.testing.assert_array_equal(
            client._gripper_closed_pose["right"], np.zeros(3)
        )
