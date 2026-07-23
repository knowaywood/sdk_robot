import sys
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np


class _Message:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


try:
    import x2robot  # noqa: F401
except ImportError:
    x2robot = types.ModuleType("x2robot")
    x2robot.Robot = _Message
    x2robot.connect = lambda _uri: None

    geometry_msgs = types.ModuleType("x2robot.geometry_msgs")
    geometry_msgs.Point = _Message
    geometry_msgs.Pose = _Message
    geometry_msgs.Quaternion = _Message

    sdk = types.ModuleType("x2robot.sdk")
    for name in [
        "GripperPosition",
        "JointPositions",
        "ManipulatorControlModeParam",
        "RobotModeParam",
    ]:
        setattr(sdk, name, _Message)
    sdk.ManipulatorControlMode = types.SimpleNamespace(
        MANIPULATOR_END_POSE=1, MANIPULATOR_JOINT_POSITIONS=2
    )
    sdk.RobotWorkMode = types.SimpleNamespace(SDK=1)

    sensor_msgs = types.ModuleType("x2robot.sensor_msgs")
    sensor_msgs.CompressedImage = type("CompressedImage", (), {})

    x2robot.geometry_msgs = geometry_msgs
    x2robot.sdk = sdk
    x2robot.sensor_msgs = sensor_msgs
    sys.modules["x2robot"] = x2robot
    sys.modules["x2robot.geometry_msgs"] = geometry_msgs
    sys.modules["x2robot.sdk"] = sdk
    sys.modules["x2robot.sensor_msgs"] = sensor_msgs

try:
    import dns.resolver  # noqa: F401
except ImportError:
    dns = types.ModuleType("dns")
    resolver = types.ModuleType("dns.resolver")
    resolver.Resolver = _Message
    resolver.NXDOMAIN = RuntimeError
    resolver.Timeout = TimeoutError
    resolver.NoAnswer = RuntimeError
    dns.resolver = resolver
    sys.modules["dns"] = dns
    sys.modules["dns.resolver"] = resolver

inference_client = types.ModuleType("x2robot_client.inference_client")
inference_client.RobotClient = _Message
sys.modules.setdefault("x2robot_client.inference_client", inference_client)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2robot_client.desktop_sdk_client import DesktopClient


def _trajectory(length, x_offset, gripper):
    trajectory = np.zeros((length, 7), dtype=np.float64)
    trajectory[:, 0] = x_offset + np.arange(length) * 0.01
    trajectory[:, -1] = gripper
    return trajectory.tolist()


class DesktopSchedulerTest(unittest.TestCase):
    def _make_client(self):
        client = DesktopClient.__new__(DesktopClient)
        client.control_mode = "end_pose"
        client.model_action_hz = 30.0
        client.exec_hz = 50.0
        client.overlap_model_steps = 3
        client._exec_lock = threading.Lock()
        client._action_buffer_l = _trajectory(53, 0.0, 0.0)
        client._action_buffer_r = _trajectory(53, 1.0, 0.0)
        client._step_idx = 10
        client._exec_seq = 120
        client._buffer_generation = 1
        client._underrun_warned = False
        return client

    def test_response_uses_request_relative_delay_and_resets_buffer_index(self):
        client = self._make_client()
        outputs = {
            "follow1_pos": _trajectory(32, 10.0, 4.5),
            "follow2_pos": _trajectory(32, 20.0, 4.5),
        }
        context = {
            "request_id": 7,
            "started_at": time.monotonic() - 0.4,
            "exec_seq": 100,
            "had_active_buffer": True,
        }

        client._execute_actions(outputs, context)

        # 32 points at 30 Hz become 53 at 50 Hz; 20 actions were committed
        # during this request, leaving 33 current/future commands.
        self.assertEqual(len(client._action_buffer_l), 33)
        self.assertEqual(client._step_idx, 0)
        self.assertEqual(client._buffer_generation, 2)
        self.assertEqual(client._action_buffer_l[0][-1], 0.0)
        self.assertEqual(client._action_buffer_l[-1][-1], 4.5)

    def test_stale_response_does_not_replace_current_buffer(self):
        client = self._make_client()
        previous_buffer = client._action_buffer_l
        outputs = {
            "follow1_pos": _trajectory(32, 10.0, 4.5),
            "follow2_pos": _trajectory(32, 20.0, 4.5),
        }
        context = {
            "request_id": 8,
            "started_at": time.monotonic() - 2.0,
            "exec_seq": 0,
            "had_active_buffer": True,
        }

        client._execute_actions(outputs, context)

        self.assertIs(client._action_buffer_l, previous_buffer)
        self.assertEqual(client._step_idx, 10)
        self.assertEqual(client._buffer_generation, 1)


if __name__ == "__main__":
    unittest.main()
