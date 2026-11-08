import cv2
import numpy as np


def frame_change(previous: np.ndarray | None, current: np.ndarray) -> float | None:
    """Diagnostic-only normalized frame difference. Never use as standalone liveness proof."""
    if previous is None or previous.shape != current.shape:
        return None
    a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return round(float(np.mean(cv2.absdiff(a, b)) / 255.0), 5)
