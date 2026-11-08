
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectionResult:
    person_count: int
    face_boxes: list[tuple[int, int, int, int]]
    person_boxes: list[tuple[int, int, int, int]]


class CVDetector:
    """
    YuNet face detector.

    Face and person boxes use the frontend-compatible
    [x, y, width, height] format.

    Person detection is supplied separately by the YOLO
    object detector in the API layer.
    """

    def __init__(self) -> None:
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "yunet"
            / "face_detection_yunet_2023mar.onnx"
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}"
            )

        self.face_detector = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            (320, 320),
            0.8,
            0.3,
            5000,
        )

        if self.face_detector is None:
            raise RuntimeError("YuNet model could not be initialized")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        height, width = frame.shape[:2]

        self.face_detector.setInputSize((width, height))

        _, faces = self.face_detector.detect(frame)

        face_boxes = []

        if faces is not None:
            for face in faces:
                x, y, box_width, box_height = face[:4]

                x = max(0, int(round(x)))
                y = max(0, int(round(y)))
                box_width = int(round(box_width))
                box_height = int(round(box_height))

                box_width = min(box_width, width - x)
                box_height = min(box_height, height - y)

                if box_width > 0 and box_height > 0:
                    face_boxes.append(
                        (x, y, box_width, box_height)
                    )

        return DetectionResult(
            person_count=0,
            face_boxes=face_boxes,
            person_boxes=[],
        )
