import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import dns.resolver
from x2robot_client.inference_client import RobotClient

logger = logging.getLogger(__name__)


class RobotClientBase(ABC):
    """Base client for robot-model interaction."""

    def __init__(
        self,
        model_address: str,
        model_port: int,
        instruction: str = "",
        max_retries: int = 3,
    ):
        self.model_address = model_address
        self.model_port = model_port
        self.instruction = instruction
        self.max_retries = max_retries

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
        self.model_output_text_buff = ""

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
    def _execute_actions(
        self, outputs: Dict[str, Any], inference_context: Any = None
    ) -> None:
        """Execute actions based on model outputs."""
        pass

    def _capture_inference_context(self) -> Any:
        """Capture state needed to align a future inference response.

        Subclasses with asynchronous action execution can override this hook.
        It is called immediately before sensor collection so response alignment
        includes both observation acquisition and model inference latency.
        """
        return None

    def get_model_output_text(self) -> None:
        """return model output text"""
        return self.model_output_text_buff

    def _update_model_output_text(self, outputs):
        """Update model output text buffer"""
        model_output_text = outputs.get("model_output_text", "")
        if model_output_text:
            self.model_output_text_buff = model_output_text

    def execute_model(self) -> None:
        """Main execution loop."""
        logger.info("Starting model execution loop...")
        try:
            while not self.action_terminator:
                if self.remote_control:
                    time.sleep(0.1)
                else:
                    # 1. Collect sensor data
                    inference_context = self._capture_inference_context()
                    st_time_1 = time.monotonic()
                    sensor_data = self._collect_sensor_data()
                    ed_time_1 = time.monotonic()

                    # 2. Inference
                    st_time_2 = time.monotonic()
                    outputs = self._inference_with_retry(sensor_data)
                    ed_time_2 = time.monotonic()

                    # 3. Execute actions
                    st_time_3 = time.monotonic()
                    self._execute_actions(outputs, inference_context)
                    ed_time_3 = time.monotonic()

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
                outputs = self.client.predict_sync(inputs)
                self._update_model_output_text(outputs)
                return outputs
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
