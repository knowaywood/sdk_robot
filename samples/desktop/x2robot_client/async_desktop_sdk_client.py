import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import transform
from x2robot.geometry_msgs import Point, Pose, Quaternion
from x2robot.sdk import GripperPosition, JointPositions
from x2robot_client.desktop_sdk_client import DesktopClient, logger
from x2robot_client.sdk_utils import (
    crossfade_trajectories,
    interpolate_trajectory,
    stitch_trajectory,
)


@dataclass(frozen=True)
class _InferenceRequest:
    request_id: int
    control_step: Optional[int]


@dataclass(frozen=True)
class _PredictionChunk:
    request: _InferenceRequest
    outputs: Optional[Dict]
    left_anchor: Optional[List[float]]
    right_anchor: Optional[List[float]]
    started_at: float
    finished_at: float
    error: Optional[Exception] = None

    @property
    def latency(self) -> float:
        return self.finished_at - self.started_at


def _next_control_deadline(
    tick_started: float, tick_finished: float, control_period: float
) -> Tuple[float, bool]:
    scheduled = tick_started + control_period
    if scheduled <= tick_finished:
        return tick_finished + control_period, True
    return scheduled, False


class AsyncDesktopClient(DesktopClient):
    """Desktop client with asynchronous inference and receding-horizon control."""

    GRIPPER_PHYSICAL_MAX = 4.5
    GRIPPER_HEARTBEAT_SECONDS = 1.0

    def __init__(
        self,
        *args,
        control_hz: float = 120.0,
        prefetch_margin: float = 0.2,
        blend_duration: float = 0.15,
        gripper_deadband: float = 0.05,
        **kwargs,
    ):
        if control_hz <= 0 or control_hz > 200:
            raise ValueError("control_hz must be in the range (0, 200]")
        if prefetch_margin < 0 or blend_duration < 0 or gripper_deadband < 0:
            raise ValueError("pipeline timing and deadband values must be non-negative")

        self.control_period = 1.0 / control_hz
        self.prefetch_margin = prefetch_margin
        self.blend_steps = int(round(blend_duration / self.control_period))
        self.gripper_deadband = gripper_deadband

        self._request_queue = queue.Queue(maxsize=1)
        self._prediction_queue = queue.Queue(maxsize=1)
        self._pipeline_stop = threading.Event()
        self._inference_thread = None
        self._inference_in_flight = False
        self._request_counter = 0
        self._discard_before_request_id = 0
        self._last_gripper_command = {"left": None, "right": None}
        self._last_gripper_sent_at = {"left": 0.0, "right": 0.0}

        super().__init__(*args, **kwargs)

    def safe_stop(self):
        pipeline_stop = getattr(self, "_pipeline_stop", None)
        if pipeline_stop is not None:
            pipeline_stop.set()
        super().safe_stop()

    @staticmethod
    def _drain_queue(target_queue):
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return

    def _start_inference_worker(self):
        self._pipeline_stop.clear()
        self._drain_queue(self._request_queue)
        self._drain_queue(self._prediction_queue)
        self._inference_in_flight = False
        if self._inference_thread is None or not self._inference_thread.is_alive():
            self._inference_thread = threading.Thread(
                target=self._inference_worker,
                name="desktop-inference-worker",
                daemon=True,
            )
            self._inference_thread.start()

    def _put_latest_prediction(self, prediction: _PredictionChunk):
        try:
            self._prediction_queue.put_nowait(prediction)
            return
        except queue.Full:
            pass

        try:
            self._prediction_queue.get_nowait()
        except queue.Empty:
            pass
        self._prediction_queue.put_nowait(prediction)

    def _inference_worker(self):
        while not self._pipeline_stop.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            started_at = time.monotonic()
            try:
                inputs = self._collect_sensor_data()
                left_anchor = inputs["state"]["follow1_pos"].tolist()
                right_anchor = inputs["state"]["follow2_pos"].tolist()
                outputs = self._inference_with_retry(inputs)
                prediction = _PredictionChunk(
                    request=request,
                    outputs=outputs,
                    left_anchor=left_anchor,
                    right_anchor=right_anchor,
                    started_at=started_at,
                    finished_at=time.monotonic(),
                )
            except Exception as exc:
                prediction = _PredictionChunk(
                    request=request,
                    outputs=None,
                    left_anchor=None,
                    right_anchor=None,
                    started_at=started_at,
                    finished_at=time.monotonic(),
                    error=exc,
                )
            self._put_latest_prediction(prediction)

    def _request_prediction(self, control_step: Optional[int]):
        if self._inference_in_flight:
            return
        self._request_counter += 1
        request = _InferenceRequest(self._request_counter, control_step)
        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            return
        self._inference_in_flight = True

    def _poll_prediction(self) -> Optional[_PredictionChunk]:
        try:
            prediction = self._prediction_queue.get_nowait()
        except queue.Empty:
            return None
        self._inference_in_flight = False
        if prediction.error is not None:
            raise RuntimeError("Asynchronous inference failed") from prediction.error
        if prediction.request.request_id <= self._discard_before_request_id:
            logger.info(
                f"Discarding invalidated prediction #{prediction.request.request_id}"
            )
            return None
        return prediction

    def _model_gripper_max(self) -> float:
        return float(getattr(self, "GRIPPER_DATA_MAX", self.GRIPPER_PHYSICAL_MAX))

    def _anchor_to_model_range(self, anchor: Optional[List[float]]):
        if anchor is None:
            return None
        converted = list(anchor)
        converted[-1] = (
            converted[-1] / self.GRIPPER_PHYSICAL_MAX * self._model_gripper_max()
        )
        return converted

    @staticmethod
    def _normalize_actions(actions) -> list:
        if actions is None:
            return []
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        return [list(action) for action in actions]

    def _prepare_action_chunk(
        self,
        outputs: Dict,
        left_anchor: Optional[List[float]],
        right_anchor: Optional[List[float]],
    ) -> List[Tuple[list, list]]:
        left_actions = self._normalize_actions(outputs.get("follow1_pos"))
        right_actions = self._normalize_actions(outputs.get("follow2_pos"))
        if not left_actions or not right_actions:
            return []

        if self.interpolate_multiplier > 1:
            left_anchor = self._anchor_to_model_range(left_anchor)
            right_anchor = self._anchor_to_model_range(right_anchor)
            if left_anchor is not None:
                left_actions.insert(0, left_anchor)
            if right_anchor is not None:
                right_actions.insert(0, right_anchor)

        left_actions = interpolate_trajectory(
            left_actions, self.interpolate_multiplier, self.control_mode
        )
        right_actions = interpolate_trajectory(
            right_actions, self.interpolate_multiplier, self.control_mode
        )
        if len(left_actions) != len(right_actions):
            logger.warning(
                "Left/right action lengths differ; truncating to the shorter chunk"
            )
        return list(zip(left_actions, right_actions))

    def _stitch_prediction(
        self,
        prediction: _PredictionChunk,
        control_step: int,
        last_command: Optional[Tuple[list, list]],
        active_actions: List[Tuple[list, list]],
    ) -> List[Tuple[list, list]]:
        prepared = self._prepare_action_chunk(
            prediction.outputs,
            prediction.left_anchor,
            prediction.right_anchor,
        )
        if not prepared:
            return []

        elapsed_steps = 0
        if prediction.request.control_step is not None:
            elapsed_steps = max(0, control_step - prediction.request.control_step)

        if elapsed_steps >= len(prepared):
            logger.warning(
                f"Discarding stale prediction #{prediction.request.request_id}: "
                f"elapsed={elapsed_steps}, chunk={len(prepared)}"
            )
            return []

        remaining = prepared[elapsed_steps:]
        if active_actions:
            left_actions = crossfade_trajectories(
                [pair[0] for pair in active_actions],
                [pair[0] for pair in remaining],
                self.blend_steps,
                self.control_mode,
            )
            right_actions = crossfade_trajectories(
                [pair[1] for pair in active_actions],
                [pair[1] for pair in remaining],
                self.blend_steps,
                self.control_mode,
            )
        else:
            left_current = last_command[0] if last_command is not None else None
            right_current = last_command[1] if last_command is not None else None
            left_actions = stitch_trajectory(
                left_current,
                [pair[0] for pair in remaining],
                0,
                self.blend_steps,
                self.control_mode,
            )
            right_actions = stitch_trajectory(
                right_current,
                [pair[1] for pair in remaining],
                0,
                self.blend_steps,
                self.control_mode,
            )

        overlap_steps = min(
            self.blend_steps, len(active_actions), len(remaining)
        )
        logger.info(
            f"Prediction #{prediction.request.request_id}: "
            f"latency={prediction.latency:.3f}s, skipped={elapsed_steps}, "
            f"crossfade={overlap_steps}, "
            f"buffered={min(len(left_actions), len(right_actions))}"
        )
        return list(zip(left_actions, right_actions))

    def _send_gripper_if_needed(self, side: str, gripper, position: float, now: float):
        previous = self._last_gripper_command[side]
        heartbeat_due = (
            now - self._last_gripper_sent_at[side] >= self.GRIPPER_HEARTBEAT_SECONDS
        )
        if (
            previous is None
            or abs(position - previous) >= self.gripper_deadband
            or heartbeat_due
        ):
            gripper.set_position(GripperPosition(position=position))
            self._last_gripper_command[side] = position
            self._last_gripper_sent_at[side] = now

    def _gripper_to_physical_range(self, value: float) -> float:
        scaled = value / self._model_gripper_max() * self.GRIPPER_PHYSICAL_MAX
        return max(0.0, min(self.GRIPPER_PHYSICAL_MAX, scaled))

    def _send_action_step(self, left_action: list, right_action: list):
        if self.control_mode == "end_pose":
            left_quat = transform.Rotation.from_euler(
                "xyz", left_action[3:6]
            ).as_quat()
            right_quat = transform.Rotation.from_euler(
                "xyz", right_action[3:6]
            ).as_quat()
            self.robot_controller.left_arm.set_end_pose(
                Pose(
                    position=Point(
                        x=float(left_action[0]),
                        y=float(left_action[1]),
                        z=float(left_action[2]),
                    ),
                    orientation=Quaternion(
                        x=float(left_quat[0]),
                        y=float(left_quat[1]),
                        z=float(left_quat[2]),
                        w=float(left_quat[3]),
                    ),
                )
            )
            self.robot_controller.right_arm.set_end_pose(
                Pose(
                    position=Point(
                        x=float(right_action[0]),
                        y=float(right_action[1]),
                        z=float(right_action[2]),
                    ),
                    orientation=Quaternion(
                        x=float(right_quat[0]),
                        y=float(right_quat[1]),
                        z=float(right_quat[2]),
                        w=float(right_quat[3]),
                    ),
                )
            )
        else:
            self.robot_controller.left_arm.set_joint_positions(
                JointPositions(positions=list(left_action[:-1]))
            )
            self.robot_controller.right_arm.set_joint_positions(
                JointPositions(positions=list(right_action[:-1]))
            )

        now = time.monotonic()
        self._send_gripper_if_needed(
            "left",
            self.robot_controller.left_gripper,
            self._gripper_to_physical_range(float(left_action[-1])),
            now,
        )
        self._send_gripper_if_needed(
            "right",
            self.robot_controller.right_gripper,
            self._gripper_to_physical_range(float(right_action[-1])),
            now,
        )

    def execute_model(self) -> None:
        """Run inference and robot control concurrently using a latest-only buffer."""
        logger.info("Starting asynchronous model execution loop...")
        logger.info(
            f"Pipeline config: control_hz={1.0 / self.control_period:.1f}, "
            f"interpolate_multiplier={self.interpolate_multiplier}, "
            f"raw_action_interval="
            f"{self.interpolate_multiplier * self.control_period:.4f}s, "
            f"prefetch_margin={self.prefetch_margin:.3f}s, "
            f"blend={self.blend_steps * self.control_period:.3f}s"
        )
        self._start_inference_worker()

        active_actions = deque()
        latency_history = deque(maxlen=20)
        last_command = None
        control_step = 0
        next_tick = time.monotonic()
        last_underflow_log = 0.0
        last_stats_log = next_tick
        deadline_misses = 0
        command_duration_history = deque(maxlen=500)
        self._request_prediction(control_step=None)

        try:
            while not self.action_terminator:
                if self.remote_control:
                    active_actions.clear()
                    self._discard_before_request_id = self._request_counter
                    prediction = self._poll_prediction()
                    if prediction is not None:
                        logger.info("Discarding prediction while remote control is active")
                    time.sleep(0.05)
                    next_tick = time.monotonic() + self.control_period
                    continue

                now = time.monotonic()
                delay = next_tick - now
                if delay > 0:
                    time.sleep(delay)
                    now = time.monotonic()
                tick_started = now

                prediction = self._poll_prediction()
                if prediction is not None:
                    latency_history.append(prediction.latency)
                    replacement = self._stitch_prediction(
                        prediction,
                        control_step,
                        last_command,
                        list(active_actions),
                    )
                    if replacement:
                        active_actions = deque(replacement)

                inference_p95 = (
                    float(np.percentile(latency_history, 95))
                    if latency_history
                    else 0.5
                )
                prefetch_steps = max(
                    1,
                    int(
                        np.ceil(
                            (inference_p95 + self.prefetch_margin)
                            / self.control_period
                        )
                    ),
                )
                if not self._inference_in_flight and (
                    not active_actions or len(active_actions) <= prefetch_steps
                ):
                    request_step = control_step if active_actions else None
                    self._request_prediction(request_step)

                if active_actions:
                    left_action, right_action = active_actions.popleft()
                    command_started = time.monotonic()
                    self._send_action_step(left_action, right_action)
                    command_duration_history.append(
                        time.monotonic() - command_started
                    )
                    last_command = (left_action, right_action)
                    control_step += 1
                elif now - last_underflow_log >= 1.0:
                    logger.warning("Action buffer underflow; holding the last command")
                    last_underflow_log = now

                if now - last_stats_log >= 5.0:
                    command_p95 = (
                        float(np.percentile(command_duration_history, 95))
                        if command_duration_history
                        else 0.0
                    )
                    logger.info(
                        f"Pipeline: buffer={len(active_actions)} "
                        f"({len(active_actions) * self.control_period:.3f}s), "
                        f"infer_p95={inference_p95:.3f}s, "
                        f"command_p95={command_p95:.4f}s, "
                        f"deadline_misses={deadline_misses}"
                    )
                    last_stats_log = now

                finished_at = time.monotonic()
                next_tick, missed_deadline = _next_control_deadline(
                    tick_started, finished_at, self.control_period
                )
                deadline_misses += int(missed_deadline)
        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, stopping...")
        except Exception as exc:
            logger.error(f"Exception in asynchronous execution loop: {exc}")
            raise
        finally:
            self.safe_stop()
