from pydantic import BaseModel, Field
from typing import Literal


class QualityResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    brightness: float
    blur_score: float
    face_width_px: int


class AnalyzeResponse(BaseModel):
    status: Literal["ok", "error"]
    person_count: int
    face_count: int
    quality: QualityResult
    pad_status: Literal["NOT_CONFIGURED", "INCONCLUSIVE", "MODEL_OUTPUT"]
    pad_score: float | None = None
    policy_action: Literal[
        "RETRY_CAPTURE", "ACTIVE_CHALLENGE_REQUIRED", "PAD_INCONCLUSIVE", "PAD_REVIEW_REQUIRED"
    ]
    challenge: str | None = None
    guidance: str
    warnings: list[str] = Field(default_factory=list)
