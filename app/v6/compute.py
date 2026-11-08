from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ComputeStatus:
    torch_available: bool
    cuda_available: bool
    gpu_name: str | None
    cuda_version: str | None
    torch_version: str | None
    onnx_available: bool
    onnx_providers: list[str]
    preferred_device: str
    errors: list[str]


class ComputeManager:
    """Reports available compute runtimes without assuming model placement."""

    def __init__(self) -> None:
        self._status = self._inspect()

    @staticmethod
    def _inspect() -> ComputeStatus:
        errors: list[str] = []

        torch_available = False
        cuda_available = False
        gpu_name = None
        cuda_version = None
        torch_version = None

        onnx_available = False
        onnx_providers: list[str] = []

        try:
            import torch

            torch_available = True
            torch_version = torch.__version__
            cuda_available = bool(torch.cuda.is_available())
            cuda_version = torch.version.cuda

            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)

        except Exception as exc:
            errors.append(f"PyTorch inspection failed: {exc}")

        try:
            import onnxruntime as ort

            onnx_available = True
            onnx_providers = ort.get_available_providers()

        except Exception as exc:
            errors.append(f"ONNX Runtime inspection failed: {exc}")

        preferred_device = "cuda" if cuda_available else "cpu"

        return ComputeStatus(
            torch_available=torch_available,
            cuda_available=cuda_available,
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            torch_version=torch_version,
            onnx_available=onnx_available,
            onnx_providers=onnx_providers,
            preferred_device=preferred_device,
            errors=errors,
        )

    def status(self) -> dict[str, Any]:
        return asdict(self._status)

    def select_torch_device(self, requested: str = "auto") -> str:
        requested = requested.lower().strip()

        if requested not in {"auto", "cuda", "cpu"}:
            raise ValueError("Device must be auto, cuda, or cpu")

        if requested == "cpu":
            return "cpu"

        if requested == "cuda":
            if not self._status.cuda_available:
                return "cpu"
            return "cuda"

        return self._status.preferred_device

    def select_onnx_providers(self) -> list[str]:
        available = self._status.onnx_providers

        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]

        return available
