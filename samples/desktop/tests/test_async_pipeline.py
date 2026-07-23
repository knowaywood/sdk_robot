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
    client.blend_steps = 2
    client._inference_clients = [object(), object()]
    client._inference_in_flight = set()
    client._request_queue = Queue(maxsize=2)
    client._prediction_queue = Queue(maxsize=4)
    client._request_counter = 0
    client._discard_before_request_id = 0
    client._last_applied_request_id = 0
    client.gripper_deadband = 0.05
    client.GRIPPER_DATA_MAX = 4.5
    client._last_gripper_command = {"left": None, "right": None}
    client._last_gripper_sent_at = {"left": 0.0, "right": 0.0}
    return client


class AsyncPipelineTest(unittest.TestCase):
    def test_control_deadline_never_catches_up_with_a_burst(self):
        next_tick, missed = _next_control_deadline(1.0, 1.015, 0.01)
        self.assertTrue(missed)
        self.assertAlmostEqual(next_tick, 1.025)

        next_tick, missed = _next_control_deadline(2.0, 2.004, 0.01)
        self.assertFalse(missed)
        self.assertAlmostEqual(next_tick, 2.01)

    def test_prediction_aligns_to_current_state_without_time_based_skip(self):
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
            last_command=([-1.0, 4.5], [-1.0, 4.5]),
            active_actions=[],
        )

        self.assertEqual(
            [pair[0][0] for pair in result], [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        )
        self.assertEqual(
            [pair[0][-1] for pair in result],
            [4.5, 4.5, 4.5, 4.5, 4.5, 0.0],
        )

    def test_alignment_uses_nearest_dual_arm_state_instead_of_elapsed_time(self):
        client = _make_client()
        prepared = [
            ([0.0, 4.5], [0.0, 4.5]),
            ([1.0, 4.5], [1.0, 4.5]),
            ([2.0, 4.5], [2.0, 4.5]),
            ([3.0, 4.5], [3.0, 4.5]),
        ]

        index = client._find_alignment_index(
            prepared, ([1.1, 4.5], [0.9, 4.5])
        )

        self.assertEqual(index, 1)

    def test_alignment_keeps_enough_actions_for_next_prediction(self):
        client = _make_client()
        prepared = [
            ([float(index), 4.5], [float(index), 4.5]) for index in range(6)
        ]

        index = client._find_alignment_index(
            prepared,
            ([5.0, 4.5], [5.0, 4.5]),
            minimum_buffer_steps=4,
        )

        self.assertEqual(index, 2)
        self.assertEqual(len(prepared[index:]), 4)

    def test_rebase_releases_joint_offset_smoothly(self):
        client = _make_client()

        result = client._rebase_actions(
            [[2.0, 4.5], [4.0, 4.5], [6.0, 0.0]],
            reference=[0.0, 4.5],
            correction_steps=3,
        )

        self.assertEqual([action[0] for action in result], [0.0, 3.0, 6.0])
        self.assertEqual([action[-1] for action in result], [4.5, 4.5, 0.0])

    def test_rebase_end_pose_starts_at_reference_and_keeps_target(self):
        client = _make_client()
        client.control_mode = "end_pose"
        actions = [
            [1.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-179.0), 4.5],
            [2.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-178.0), 4.5],
            [3.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-177.0), 0.0],
        ]
        reference = [0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(179.0), 4.5]

        result = np.asarray(client._rebase_actions(actions, reference, 3))
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

    def test_gripper_deadband_suppresses_duplicate_rpc_calls(self):
        client = _make_client()
        gripper = _RecordingGripper()

        client._send_gripper_if_needed("left", gripper, 2.0, now=1.0)
        client._send_gripper_if_needed("left", gripper, 2.02, now=1.1)
        client._send_gripper_if_needed("left", gripper, 2.2, now=1.2)
        client._send_gripper_if_needed("left", gripper, 2.2, now=2.3)

        self.assertEqual(gripper.positions, [2.0, 2.2, 2.2])
