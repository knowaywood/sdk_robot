import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2robot_client.async_desktop_sdk_client import (
    AsyncDesktopClient,
    _InferenceRequest,
    _PredictionChunk,
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
    client.gripper_deadband = 0.05
    client._last_gripper_command = {"left": None, "right": None}
    client._last_gripper_sent_at = {"left": 0.0, "right": 0.0}
    return client


class AsyncPipelineTest(unittest.TestCase):
    def test_prediction_skips_steps_executed_while_inference_was_running(self):
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
        )

        self.assertEqual([pair[0][0] for pair in result], [0.0, 1.0, 1.5, 2.0])
        self.assertEqual([pair[0][-1] for pair in result], [4.5, 4.5, 4.5, 0.0])

    def test_gripper_deadband_suppresses_duplicate_rpc_calls(self):
        client = _make_client()
        gripper = _RecordingGripper()

        client._send_gripper_if_needed("left", gripper, 2.0, now=1.0)
        client._send_gripper_if_needed("left", gripper, 2.02, now=1.1)
        client._send_gripper_if_needed("left", gripper, 2.2, now=1.2)
        client._send_gripper_if_needed("left", gripper, 2.2, now=2.3)

        self.assertEqual(gripper.positions, [2.0, 2.2, 2.2])
