import cv2
import numpy as np
from app.schemas import QualityResult


def assess_quality(
    frame: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]],
    face_width_px: int | None = None,
) -> QualityResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    detected_face_width = max(
        (int(box[2]) for box in face_boxes),
        default=0,
    )

    # An optional original-image width preserves the existing
    # 90-pixel threshold when analysis uses a resized copy.
    effective_face_width = (
        detected_face_width
        if face_width_px is None
        else max(0, int(face_width_px))
    )

    reasons: list[str] = []

    if len(face_boxes) == 0:
        reasons.append("No face detected")
    elif len(face_boxes) > 1:
        reasons.append("More than one face detected")

    if effective_face_width and effective_face_width < 90:
        reasons.append("Move closer to the camera")

    if brightness < 45:
        reasons.append("Scene is too dark")
    elif brightness > 215:
        reasons.append("Scene is overexposed")

    if blur_score < 35:
        reasons.append("Image appears blurred; hold still")

    return QualityResult(
        passed=(len(reasons) == 0),
        reasons=reasons,
        brightness=round(brightness, 2),
        blur_score=round(blur_score, 2),
        face_width_px=effective_face_width,
    )
