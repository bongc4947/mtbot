"""GPU detection and benchmarking for optional DirectML / ONNX acceleration."""
import time
import numpy as np
from utils.logger import get_logger

log = get_logger("gpu_utils")


def detect_onnx_directml() -> bool:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "DmlExecutionProvider" in providers
    except Exception:
        return False


def detect_torch_directml() -> bool:
    try:
        import torch_directml  # noqa
        return True
    except Exception:
        return False


def benchmark_device(device_name: str, size: int = 2000, iterations: int = 50) -> float:
    """Return throughput (samples/sec). Higher = faster."""
    try:
        if device_name == "cpu":
            a = np.random.randn(size, size).astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(iterations):
                np.dot(a, a)
            elapsed = time.perf_counter() - t0
        else:
            import torch
            import torch_directml
            dml = torch_directml.device()
            a = torch.randn(size, size, device=dml)
            t0 = time.perf_counter()
            for _ in range(iterations):
                torch.mm(a, a)
            elapsed = time.perf_counter() - t0
        return (size * size * iterations) / elapsed
    except Exception as e:
        log.warning(f"Benchmark failed for {device_name}: {e}")
        return 0.0


class AccelerationContext:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.use_gpu = False
        self.provider = "cpu"
        self._setup()

    def _setup(self):
        if not self.cfg.get("enabled", True):
            log.info("GPU disabled by config — using CPU")
            return

        has_dml_ort = detect_onnx_directml()
        has_dml_torch = detect_torch_directml()

        if not has_dml_ort and not has_dml_torch:
            log.info("No DirectML providers found — using CPU")
            return

        if self.cfg.get("benchmark_on_startup", True):
            cpu_score = benchmark_device("cpu")
            gpu_score = benchmark_device("dml") if has_dml_torch else 0.0
            log.info(f"Benchmark — CPU: {cpu_score:.0f}  GPU(DML): {gpu_score:.0f}")
            if gpu_score > cpu_score * 1.2:
                self.use_gpu = True
                self.provider = "dml_ort" if has_dml_ort else "dml_torch"
            else:
                log.info("GPU not faster than CPU — staying on CPU")
        else:
            if has_dml_ort:
                self.use_gpu = True
                self.provider = "dml_ort"

        log.info(f"Acceleration: provider={self.provider}, gpu={self.use_gpu}")

    def get_onnx_providers(self) -> list:
        if self.use_gpu and self.provider == "dml_ort":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
