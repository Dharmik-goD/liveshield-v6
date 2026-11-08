import numpy as np
from app.quality import assess_quality


def test_no_face_fails_quality_gate():
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    result = assess_quality(frame, [])
    assert not result.passed
    assert "No face detected" in result.reasons
