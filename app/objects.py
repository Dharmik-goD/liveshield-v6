import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class ObjectDetection:
    label: str
    confidence: float
    box: list[int]

class ObjectDetector:
    """Optional local YOLO detector; never downloads weights automatically."""
    def __init__(self):
        self.model = None
        self.error = None
        self.model_path = self._configured_model_path()
        self.device = "cpu"
        self._load_attempted = False

    @staticmethod
    def _configured_model_path() -> Path:
        configured = Path(os.getenv("OBJECT_MODEL", "yolov8n.pt"))
        if not configured.is_absolute():
            configured = Path(__file__).resolve().parent.parent / configured
        return configured.resolve()

    def _load_model(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.model_path.is_file():
            self.error = f"Local object model not found: {self.model_path}"
            return
        try:
            import torch
            from ultralytics import YOLO

            requested_device = os.getenv("OBJECT_DEVICE", "auto").lower()
            self.device = (
                "cuda" if requested_device in {"auto", "cuda"} and torch.cuda.is_available()
                else "cpu"
            )
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            self.error = str(exc)

    @property
    def ready(self):
        return self.model is not None

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        self._load_model()
        if not self.model:
            return []
        results = self.model.predict(source=frame, conf=0.35, imgsz=640,
                                     verbose=False, device=self.device)
        found = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                found.append(ObjectDetection(
                    label=str(result.names[class_id]),
                    confidence=round(float(box.conf.item()), 4),
                    box=[x1, y1, x2, y2]))
        return found
