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
from x2robot_client.inference_client import RobotClient
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
    INITIAL_INFERENCE_LATENCY_SECONDS = 1.2

    def __init__(
        self,
        *args,
        control_hz: float = 110.0,
        prefetch_margin: float = 0.1,
        blend_duration: float = 0.3,
        gripper_deadband: float = 0.05,
        inference_workers: int = 1,
        **kwargs,
    ):
        if control_hz <= 0 or control_hz > 200:
            raise ValueError("control_hz must be in the range (0, 200]")
        if prefetch_margin < 0 or blend_duration < 0 or gripper_deadband < 0:
            raise ValueError("pipeline timing and deadband values must be non-negative")
        if inference_workers < 1 or inference_workers > 4:
            raise ValueError("inference_workers must be in the range [1, 4]")

        self.control_period = 1.0 / control_hz
        self.prefetch_margin = prefetch_margin
        self.blend_steps = int(round(blend_duration / self.control_period))
        self.gripper_deadband = gripper_deadband
        self.inference_workers = inference_workers

        self._request_queue = queue.Queue(maxsize=inference_workers)
        self._prediction_queue = queue.Queue(maxsize=2 * inference_workers)
        self._pipeline_stop = threading.Event()
        self._inference_threads = []
        self._inference_clients = []
        self._sensor_lock = threading.Lock()
        self._inference_in_flight = set()
        self._request_counter = 0
        self._discard_before_request_id = 0
        self._last_applied_request_id = 0
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
        self._inference_in_flight.clear()

        clients = [self.client]
        metadata = getattr(self.client, "metadata", None) or {}
        worker_limit = self.inference_workers
        if worker_limit > 1 and not metadata.get("batch_enabled", False):
            logger.warning(
                "Model server does not advertise batch support; using one "
                "inference worker"
            )
            worker_limit = 1

        for worker_index in range(1, worker_limit):
            try:
                client = RobotClient(uri=self.uri)
                client.connect_sync()
                clients.append(client)
            except Exception as exc:
                logger.warning(
                    f"Could not start inference worker {worker_index + 1}: {exc}; "
                    f"continuing with {len(clients)} worker(s)"
                )
                break

        self._inference_clients = clients
        self._inference_threads = []
        for worker_index, client in enumerate(clients, start=1):
            thread = threading.Thread(
                target=self._inference_worker,
                args=(client,),
                name=f"desktop-inference-worker-{worker_index}",
                daemon=True,
            )
            thread.start()
            self._inference_threads.append(thread)

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

    def _predict_with_retry(self, client: RobotClient, inputs: Dict) -> Dict:
        for attempt in range(self.max_retries):
            try:
                outputs = client.predict_sync(inputs)
                self._update_model_output_text(outputs)
                return outputs
            except Exception as exc:
                logger.warning(
                    f"Inference failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"{exc}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)
        raise RuntimeError(f"Model inference failed after {self.max_retries} attempts")

    def _inference_worker(self, client: RobotClient):
        while not self._pipeline_stop.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            started_at = time.monotonic()
            try:
                # Robot SDK sensor reads are serialized; model calls still overlap.
                with self._sensor_lock:
                    inputs = self._collect_sensor_data()
                left_anchor = inputs["state"]["follow1_pos"].tolist()
                right_anchor = inputs["state"]["follow2_pos"].tolist()
                outputs = self._predict_with_retry(client, inputs)
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
        if len(self._inference_in_flight) >= len(self._inference_clients):
            return False
        self._request_counter += 1
        request = _InferenceRequest(self._request_counter, control_step)
        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            return False
        self._inference_in_flight.add(request.request_id)
        return True

    def _poll_prediction(self) -> Optional[_PredictionChunk]:
        try:
            prediction = self._prediction_queue.get_nowait()
        except queue.Empty:
            return None
        self._inference_in_flight.discard(prediction.request.request_id)
        if prediction.request.request_id <= max(
            self._discard_before_request_id, self._last_applied_request_id
        ):
            logger.info(
                f"Discarding stale prediction #{prediction.request.request_id}"
            )
            return None
        if prediction.error is not None:
            raise RuntimeError("Asynchronous inference failed") from prediction.error
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
        last_command: Optional[Tuple[list, list]],
        active_actions: List[Tuple[list, list]],
        minimum_buffer_steps: int = 0,
    ) -> List[Tuple[list, list]]:
        prepared = self._prepare_action_chunk(
            prediction.outputs,
            prediction.left_anchor,
            prediction.right_anchor,
        )
        if not prepared:
            return []

        reference = active_actions[0] if active_actions else last_command
        alignment_index = self._find_alignment_index(
            prepared, reference, minimum_buffer_steps
        )
        remaining = prepared[alignment_index:]
        rebased = reference is not None and minimum_buffer_steps > 0
        handoff_error = self._format_handoff_error(remaining[0], reference)
        if rebased:
            left_remaining = self._shift_actions_to_reference(
                [pair[0] for pair in remaining],
                reference[0],
            )
            right_remaining = self._shift_actions_to_reference(
                [pair[1] for pair in remaining],
                reference[1],
            )
            remaining = list(zip(left_remaining, right_remaining))
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
            f"latency={prediction.latency:.3f}s, aligned={alignment_index}, "
            f"crossfade={overlap_steps}, "
            f"rebased={str(rebased).lower()}, "
            f"handoff_error={handoff_error}, "
            f"buffered={min(len(left_actions), len(right_actions))}"
        )
        return list(zip(left_actions, right_actions))

    def _format_handoff_error(
        self,
        predicted: Tuple[list, list],
        reference: Optional[Tuple[list, list]],
    ) -> str:
        if reference is None:
            return "initial"

        predicted_left = np.asarray(predicted[0], dtype=float)
        predicted_right = np.asarray(predicted[1], dtype=float)
        reference_left = np.asarray(reference[0], dtype=float)
        reference_right = np.asarray(reference[1], dtype=float)
        if self.control_mode == "end_pose":
            left_position = np.linalg.norm(predicted_left[:3] - reference_left[:3])
            right_position = np.linalg.norm(
                predicted_right[:3] - reference_right[:3]
            )
            left_rotation = (
                transform.Rotation.from_euler("xyz", reference_left[3:6]).inv()
                * transform.Rotation.from_euler("xyz", predicted_left[3:6])
            ).magnitude()
            right_rotation = (
                transform.Rotation.from_euler("xyz", reference_right[3:6]).inv()
                * transform.Rotation.from_euler("xyz", predicted_right[3:6])
            ).magnitude()
            return (
                f"pos={left_position:.4f}/{right_position:.4f}m,"
                f"rot={np.degrees(left_rotation):.1f}/"
                f"{np.degrees(right_rotation):.1f}deg"
            )

        left_joints = np.linalg.norm(predicted_left[:-1] - reference_left[:-1])
        right_joints = np.linalg.norm(predicted_right[:-1] - reference_right[:-1])
        return f"joints={left_joints:.4f}/{right_joints:.4f}rad"

    def _shift_actions_to_reference(
        self,
        actions: List[list],
        reference: list,
    ) -> List[list]:
        """Move a trajectory to the handoff state while preserving its increments."""
        if not actions:
            return [list(action) for action in actions]

        corrected = np.asarray(actions, dtype=float).copy()
        reference_array = np.asarray(reference, dtype=float)

        if self.control_mode == "end_pose":
            position_offset = reference_array[:3] - corrected[0, :3]
            corrected[:, :3] += position_offset
            reference_rotation = transform.Rotation.from_euler(
                "xyz", reference_array[3:6]
            )
            initial_rotation = transform.Rotation.from_euler(
                "xyz", corrected[0, 3:6]
            )
            rotation_offset = reference_rotation * initial_rotation.inv()
            target_rotations = transform.Rotation.from_euler(
                "xyz", corrected[:, 3:6]
            )
            corrected[:, 3:6] = (
                rotation_offset * target_rotations
            ).as_euler("xyz")
        else:
            joint_offset = reference_array[:-1] - corrected[0, :-1]
            corrected[:, :-1] += joint_offset

        return corrected.tolist()

    def _find_alignment_index(
        self,
        prepared: List[Tuple[list, list]],
        reference: Optional[Tuple[list, list]],
        minimum_buffer_steps: int = 0,
    ) -> int:
        if reference is None or len(prepared) <= 1:
            return 0

        minimum_remaining = max(1, self.blend_steps, minimum_buffer_steps)
        candidate_count = max(1, len(prepared) - minimum_remaining + 1)
        left = np.asarray([pair[0] for pair in prepared[:candidate_count]])
        right = np.asarray([pair[1] for pair in prepared[:candidate_count]])
        left_reference = np.asarray(reference[0])
        right_reference = np.asarray(reference[1])

        if self.control_mode == "end_pose":
            position_cost = np.linalg.norm(
                left[:, :3] - left_reference[:3], axis=1
            ) + np.linalg.norm(right[:, :3] - right_reference[:3], axis=1)
            left_reference_rotation = transform.Rotation.from_euler(
                "xyz", left_reference[3:6]
            )
            right_reference_rotation = transform.Rotation.from_euler(
                "xyz", right_reference[3:6]
            )
            left_angle = (
                left_reference_rotation.inv()
                * transform.Rotation.from_euler("xyz", left[:, 3:6])
            ).magnitude()
            right_angle = (
                right_reference_rotation.inv()
                * transform.Rotation.from_euler("xyz", right[:, 3:6])
            ).magnitude()
            cost = position_cost + 0.1 * (left_angle + right_angle)
        else:
            cost = np.linalg.norm(
                left[:, :-1] - left_reference[:-1], axis=1
            ) + np.linalg.norm(right[:, :-1] - right_reference[:-1], axis=1)

        return int(np.argmin(cost))

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
        """Run staggered inference and receding-horizon robot control."""
        logger.info("Starting asynchronous model execution loop...")
        self._start_inference_worker()
        worker_count = len(self._inference_clients)
        logger.info(
            f"Pipeline config: control_hz={1.0 / self.control_period:.1f}, "
            f"interpolate_multiplier={self.interpolate_multiplier}, "
            f"raw_action_interval="
            f"{self.interpolate_multiplier * self.control_period:.4f}s, "
            f"prefetch_margin={self.prefetch_margin:.3f}s, "
            f"blend={self.blend_steps * self.control_period:.3f}s, "
            f"inference_workers={worker_count}"
        )

        active_actions = deque()
        latency_history = deque(maxlen=20)
        last_command = None
        pending_prediction = None
        control_step = 0
        next_tick = time.monotonic()
        last_request_at = next_tick
        last_underflow_log = 0.0
        last_stats_log = next_tick
        deadline_misses = 0
        command_duration_history = deque(maxlen=500)
        self._request_prediction(control_step=None)

        try:
            while not self.action_terminator:
                if self.remote_control:
                    active_actions.clear()
                    pending_prediction = None
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
                    pending_prediction = prediction

                inference_p95 = (
                    float(np.percentile(latency_history, 95))
                    if latency_history
                    else self.INITIAL_INFERENCE_LATENCY_SECONDS
                )
                if pending_prediction is not None and (
                    not active_actions or len(active_actions) <= self.blend_steps
                ):
                    minimum_buffer_steps = int(
                        np.ceil(
                            (inference_p95 / worker_count + self.prefetch_margin)
                            / self.control_period
                        )
                    )
                    replacement = self._stitch_prediction(
                        pending_prediction,
                        last_command,
                        list(active_actions),
                        minimum_buffer_steps,
                    )
                    if replacement:
                        active_actions = deque(replacement)
                        self._last_applied_request_id = (
                            pending_prediction.request.request_id
                        )
                    pending_prediction = None

                request_interval = max(
                    self.control_period,
                    inference_p95 / worker_count,
                )
                request_due = worker_count == 1 or (
                    now - last_request_at >= request_interval
                )
                if len(self._inference_in_flight) < worker_count and request_due:
                    request_step = control_step if active_actions else None
                    if self._request_prediction(request_step):
                        last_request_at = now

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
                        f"result_interval={request_interval:.3f}s, "
                        f"in_flight={len(self._inference_in_flight)}, "
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
