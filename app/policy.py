from app.schemas import QualityResult
from app.challenges import choose_challenge


def decide(quality: QualityResult, pad_configured: bool) -> tuple[str, str | None, str]:
    if not quality.passed:
        return "RETRY_CAPTURE", None, "; ".join(quality.reasons)
    if not pad_configured:
        return (
            "ACTIVE_CHALLENGE_REQUIRED",
            choose_challenge(),
            "Capture quality passed. PAD model is not configured; complete the challenge, "
            "but the demo will still not declare the presentation bona fide.",
        )
    return (
        "PAD_REVIEW_REQUIRED",
        choose_challenge(),
        "Model output requires calibrated thresholding and sequence-level challenge verification.",
    )
