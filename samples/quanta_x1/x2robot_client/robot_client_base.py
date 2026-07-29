import logging
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
        """Main execution loop."""
        logger.info("Starting model execution loop...")
        try:
            while not self.action_terminator:
                if self.remote_control:
                    time.sleep(0.1)
                else:
                    # 1. Collect sensor data
                    st_time_1 = time.time()
                    sensor_data = self._collect_sensor_data()
                    ed_time_1 = time.time()

                    # 2. Inference
                    st_time_2 = time.time()
                    outputs = self._inference_with_retry(sensor_data)
                    ed_time_2 = time.time()

                    # 3. Decode msgpack-numpy arrays before any processing
                    raw_decoded = decode_outputs(outputs)
                    if self.rtc_enabled:
                        # Server-side RTC: server already did three-zone blending
                        # Client only needs gripper protection
                        outputs = restore_gripper_in_outputs(raw_decoded, raw_decoded)
                    elif self.rtc_blend:
                        # Client-side three-zone RTC blending
                        outputs = rtc_blend_outputs(
                            self.prev_outputs, raw_decoded,
                            self.rtc_delay_steps, self.rtc_overlap_steps,
                        )
                        outputs = restore_gripper_in_outputs(outputs, raw_decoded)
                    elif self.smooth_chunks:
                        # Cosine-decay boundary blending (original mode)
                        outputs = smooth_chunk_boundary(
                            self.prev_outputs, raw_decoded, self.blend_steps
                        )
                        outputs = restore_gripper_in_outputs(outputs, raw_decoded)
                    else:
                        outputs = raw_decoded
                    self.prev_outputs = outputs

                    # 4. Execute actions
                    st_time_3 = time.time()
                    self._execute_actions(outputs)
                    ed_time_3 = time.time()

                    logger.info(
                        f"Data: {ed_time_1 - st_time_1:.4f}s, "
                        f"Infer: {ed_time_2 - st_time_2:.4f}s, "
                        f"Exec: {ed_time_3 - st_time_3:.4f}s"
                    )

        except KeyboardInterrupt:
            logger.info("Received Ctrl+C, stopping...")
            self.safe_stop()
        except Exception as e:
            logger.error(f"Exception in execution loop: {e}")
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

    def __del__(self) -> None:
        self.safe_stop()
