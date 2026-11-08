from app.policy import decide
from app.schemas import QualityResult


def quality(passed=True):
    return QualityResult(passed=passed, reasons=[] if passed else ["No face detected"],
                         brightness=120, blur_score=100, face_width_px=120)


def test_bad_quality_requires_retry():
    action, challenge, _ = decide(quality(False), False)
    assert action == "RETRY_CAPTURE"
    assert challenge is None


def test_missing_pad_never_accepts():
    action, challenge, _ = decide(quality(True), False)
    assert action == "ACTIVE_CHALLENGE_REQUIRED"
    assert challenge in {
        "Please blink naturally",
        "Please slowly turn your head to the left",
        "Please slowly turn your head to the right",
    }
