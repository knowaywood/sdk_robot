import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import dns.resolver
from x2robot_client.inference_client import RobotClient
from x2robot_client.rtc_inference import (
    decode_outputs,
    restore_gripper_in_outputs,
    smooth_chunk_boundary,
    rtc_blend_outputs,
)

logger = logging.getLogger(__name__)


class ActionBuffer:
    """Thread-safe double buffer for async inference.

    Inference thread writes completed chunks, execution thread
    reads the latest available chunk without blocking each other.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._has_new = threading.Event()
        self._front: Optional[Dict[str, Any]] = None
        self._back: Optional[Dict[str, Any]] = None

    def write(self, actions: Dict[str, Any]) -> None:
        """Write a new action chunk (called by inference thread)."""
        with self._lock:
            self._back = actions
            self._has_new.set()

    def read(self) -> Optional[Dict[str, Any]]:
        """Read the latest available chunk (called by execution thread).
        Returns None if no new chunk since last read.
        """
        with self._lock:
            if self._back is not None:
                self._front = self._back
                self._back = None
                self._has_new.clear()
            return self._front

    def wait(self, timeout: float = 0.01) -> bool:
        """Wait until a new chunk is available. Returns True if new data."""
        return self._has_new.wait(timeout)

    def clear(self) -> None:
        """Drop both buffers (called on reset/stop)."""
        with self._lock:
            self._front = None
            self._back = None
            self._has_new.clear()


class RobotClientBase(ABC):
    """Base client for robot-model interaction."""

    def __init__(
        self,
        model_address: str,
        model_port: int,
        instruction: str = "",
        max_retries: int = 3,
        smooth_chunks: bool = False,
        blend_steps: int = 3,
        rtc_enabled: bool = False,
        rtc_delay_steps: int = 0,
        rtc_overlap_steps: int = 3,
        rtc_blend: bool = False,
        async_mode: bool = False,
    ):
        self.model_address = model_address
        self.model_port = model_port
        self.instruction = instruction
        self.max_retries = max_retries
        self.smooth_chunks = smooth_chunks
        self.blend_steps = blend_steps
        self.rtc_enabled = rtc_enabled
        self.rtc_delay_steps = rtc_delay_steps
        self.rtc_overlap_steps = rtc_overlap_steps
        self.rtc_blend = rtc_blend
        self.async_mode = async_mode

        if not self.ip_address(model_address):
            try:
                self.model_address = self.resolve_dns(model_address)

            except ValueError as e:
                print(f"DNS Resolve {self.model_address} Error: {e}")
                exit(1)

        self.uri = f"ws://{self.model_address}:{self.model_port}"
        self.action_terminator = True
        self.remote_control = False
        self.client: Optional[RobotClient] = None
        self.prev_outputs: Optional[Dict[str, Any]] = None
        self._action_buffer = ActionBuffer()
        self._inference_thread: Optional[threading.Thread] = None

        # Initialize components
        self._init_inference_client()
        self._init_robot_controller()

    def ip_address(self, ip_str: str) -> bool:
        parts = ip_str.split(".")
        if len(parts) != 4:
            return False
        for part in parts:
            if not part.isdigit():
                return False
            num = int(part)
            if num < 0 or num > 255:
                return False
        return True

    def resolve_dns(self, hostname, dns_server="10.43.0.10"):
        try:
            resolver = dns.resolver.Resolver()
            resolver.nameservers = [dns_server]

            answer = resolver.resolve(hostname, "A")
            ipv4 = answer[0].address
            print(f"Resolution successful! IPv4 (A record) via {dns_server}: {ipv4}")
            return ipv4

        except dns.resolver.NXDOMAIN:
            print(f"Domain does not exist: {hostname}")
        except dns.resolver.Timeout:
            print("DNS query timed out")
        except dns.resolver.NoAnswer:
            print(f"No A record found: {hostname}")
        except Exception as e:
            print(f"DNS resolution failed: {e}")

    @abstractmethod
    def _init_robot_controller(self) -> None:
        """Initialize robot controller (implemented by subclasses)."""
        pass

    def _init_inference_client(self) -> None:
        """Initialize inference client."""
        try:
            self.client = RobotClient(uri=self.uri)
            self.client.connect_sync()
            logger.info(f"Connected to model server at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to model server: {e}")
            raise

    @abstractmethod
    def _collect_sensor_data(self) -> Dict[str, Any]:
        """Collect sensor data for inference."""
        pass

    @abstractmethod
    def _execute_actions(self, outputs: Dict[str, Any]) -> None:
        """Execute actions based on model outputs."""
        pass

    def execute_model(self) -> None:
        """Main execution loop.

        In sync mode (default): collect → infer → execute (blocking).
        In async mode:       inference runs in a background thread,
                             execution thread picks up the latest chunk.
        """
        if self.async_mode:
            self._execute_model_async()
        else:
            self._execute_model_sync()

    def _execute_model_sync(self) -> None:
        """Synchronous execution loop (original behavior)."""
        logger.info("Starting sync execution loop...")
        try:
            while not self.action_terminator:
                if self.remote_control:
                    time.sleep(0.1)
                    continue

                st_data = time.time()
                sensor_data = self._collect_sensor_data()
                t_data = time.time() - st_data

                st_infer = time.time()
                outputs = self._inference_with_retry(sensor_data)
                t_infer = time.time() - st_infer

                raw_decoded = decode_outputs(outputs)
                blended = self._blend_outputs(raw_decoded)

                st_exec = time.time()
                self._execute_actions(blended)
                t_exec = time.time() - st_exec

                logger.info(
                    f"Data: {t_data:.4f}s, Infer: {t_infer:.4f}s, Exec: {t_exec:.4f}s"
                )

        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, stopping...")
            self.safe_stop()
        except Exception as e:
            logger.error(f"Exception in execution loop: {e}")
            self.safe_stop()
            raise

    def _execute_model_async(self) -> None:
        """Asynchronous execution loop with background inference thread."""
        logger.info("Starting async execution loop...")
        self._action_buffer.clear()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True,
            name="async-inference",
        )
        self._inference_thread.start()

        try:
            while not self.action_terminator:
                if self.remote_control:
                    time.sleep(0.1)
                    continue

                # Wait for first chunk or check for new data
                chunk = self._action_buffer.read()
                if chunk is None:
                    if not self._action_buffer.wait(timeout=0.5):
                        continue
                    chunk = self._action_buffer.read()
                    if chunk is None:
                        continue

                st_exec = time.time()
                self._execute_actions(chunk)
                t_exec = time.time() - st_exec

                logger.debug(f"Async exec: {t_exec:.4f}s")

        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, stopping...")
            self.safe_stop()
        except Exception as e:
            logger.error(f"Exception in async execution loop: {e}")
            self.safe_stop()
            raise

    def _inference_with_retry(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Inference with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                return self.client.predict_sync(inputs)
            except Exception as e:
                logger.warning(
                    f"Inference failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)
        raise RuntimeError(f"Model inference failed after {self.max_retries} attempts")

    def _blend_outputs(self, raw_decoded: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configured blending strategy to raw model outputs."""
        if self.rtc_enabled:
            out = restore_gripper_in_outputs(raw_decoded, raw_decoded)
        elif self.rtc_blend:
            out = rtc_blend_outputs(
                self.prev_outputs, raw_decoded,
                self.rtc_delay_steps, self.rtc_overlap_steps,
            )
            out = restore_gripper_in_outputs(out, raw_decoded)
        elif self.smooth_chunks:
            out = smooth_chunk_boundary(
                self.prev_outputs, raw_decoded, self.blend_steps
            )
            out = restore_gripper_in_outputs(out, raw_decoded)
        else:
            out = raw_decoded
        self.prev_outputs = out
        return out

    def _inference_loop(self) -> None:
        """Background inference thread for async mode.
        Continuously collects sensor data, runs inference, blends,
        and writes to the action buffer.
        """
        logger.info("Async inference thread started")
        try:
            while not self.action_terminator:
                sensor_data = self._collect_sensor_data()
                outputs = self._inference_with_retry(sensor_data)
                raw_decoded = decode_outputs(outputs)
                blended = self._blend_outputs(raw_decoded)
                self._action_buffer.write(blended)
        except Exception as e:
            logger.error(f"Inference thread exception: {e}")
        finally:
            logger.info("Async inference thread stopped")

    def start_control(self) -> None:
        """Start control loop (enable model execution)."""
        self.action_terminator = False

    def stop_control(self) -> None:
        """Stop control loop (perform a safe stop)."""
        self.safe_stop()

    def start_remote_control(self) -> None:
        """Enable remote control mode."""
        self.remote_control = True

    def stop_remote_control(self) -> None:
        """Disable remote control mode."""
        self.remote_control = False

    def safe_stop(self) -> None:
        """Perform a safe stop procedure (sets action_terminator)."""
        logger.info("Executing safe stop...")
        self.action_terminator = True
        if self._inference_thread is not None and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5)
            logger.info("Inference thread joined")

    def __del__(self) -> None:
        self.safe_stop()
