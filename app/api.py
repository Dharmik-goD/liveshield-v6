from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.detection import CVDetector
from app.quality import assess_quality
from app.policy import decide
from app.objects import ObjectDetector
from app.vision.captioner import SceneCaptioner
from app.vision.expressions import FacialExpressionAnalyzer
from app.pad.onnx_engine import ONNXPADEngine


router = APIRouter(prefix="/api")

detector = CVDetector()
object_detector = ObjectDetector()
scene_captioner = SceneCaptioner()
expression_analyzer = FacialExpressionAnalyzer()

PAD_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "pad"
    / "minifasnet_v2.onnx"
)

pad_engine = None
pad_model_error = None

try:
    pad_engine = ONNXPADEngine(str(PAD_MODEL_PATH))
except Exception as exc:
    pad_model_error = str(exc)


@router.get("/info")
def info():
    return {
        "project": "MOSIP LiveShield V5",
        "person_detector": "YOLO",
        "face_detector": "YuNet",
        "pad_model_configured": pad_engine is not None,
        "pad_status": (
            "EXPERIMENTAL_NOT_VALIDATED"
            if pad_engine is not None
            else "NOT_CONFIGURED"
        ),
        "pad_model": "MiniFASNet V2 ONNX",
        "pad_model_error": pad_model_error,
        "pad_execution_provider": "CPUExecutionProvider",
        "object_detector": "READY" if object_detector.ready else "NOT_READY",
        "object_model_error": (
            object_detector.error if not object_detector.ready else None
        ),
        "expression_analyzer": (
            "READY" if expression_analyzer.model is not None else "NOT_READY"
        ),
        "expression_model_error": expression_analyzer.error,
        "scene_captioner": (
            "READY" if scene_captioner.ready else "FALLBACK_MODE"
        ),
        "scene_model": scene_captioner.model_id,
        "storage": "disabled",
    }


@router.post("/analyze")
async def analyze(frame: UploadFile = File(...)):
    if frame.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG or WebP frame",
        )

    raw = await frame.read()

    if not raw or len(raw) > 8_000_000:
        raise HTTPException(
            status_code=400,
            detail="Empty frame or frame exceeds 8 MB",
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image",
        )

    detections = detector.detect(image)
    expressions = expression_analyzer.analyze(
        image,
        detections.face_boxes,
    )

    objects = object_detector.detect(image)
    person_boxes = []

    for obj in objects:
        if obj.label.lower() == "person":
            x1, y1, x2, y2 = obj.box

            person_boxes.append((
                x1,
                y1,
                max(0, x2 - x1),
                max(0, y2 - y1),
            ))

    scene = scene_captioner.describe(
        image,
        [obj.label for obj in objects],
    )

    quality = assess_quality(image, detections.face_boxes)
    action, challenge, guidance = decide(quality, False)

    pad_status = "NOT_CONFIGURED"
    pad_score = None
    pad_label = "INCONCLUSIVE"
    pad_probabilities = None

    if pad_engine is not None:
        if len(detections.face_boxes) == 0:
            pad_status = "NO_FACE"

        elif len(detections.face_boxes) > 1:
            pad_status = "MULTIPLE_FACES"

        elif not quality.passed:
            pad_status = "QUALITY_REJECTED"

        else:
            try:
                result = pad_engine.predict(
                    image,
                    detections.face_boxes[0],
                )

                pad_score = result.score
                pad_label = result.label
                pad_probabilities = result.probabilities

                pad_status = (
                    "EXPERIMENTAL_MODEL_OUTPUT"
                    if result.score is not None
                    else "INCONCLUSIVE"
                )

            except Exception:
                pad_status = "INFERENCE_ERROR"
                pad_label = "INCONCLUSIVE"

    return {
        "status": "ok",
        "person_count": len(person_boxes),
        "person_boxes": [list(box) for box in person_boxes],
        "face_count": len(detections.face_boxes),
        "face_boxes": [list(box) for box in detections.face_boxes],

        "facial_expressions": expressions.get("faces", []),
        "expression_analysis_status": expressions.get(
            "status", "unavailable"
        ),
        "expression_analysis_error": expressions.get("error"),

        "objects": [
            {
                "label": obj.label,
                "confidence": obj.confidence,
                "box": obj.box,
            }
            for obj in objects
        ],
        "object_detection_status": (
            "READY" if object_detector.ready else "NOT_READY"
        ),
        "object_model_error": (
            object_detector.error if not object_detector.ready else None
        ),

        "scene_description": scene.caption,
        "scene_description_engine": scene.engine,
        "scene_model_ready": scene.ready,
        "scene_description_note": scene.note,

        "quality": quality.model_dump(),

        "pad_status": pad_status,
        "pad_score": pad_score,
        "pad_label": pad_label,
        "pad_probabilities": pad_probabilities,
        "pad_model_validated": False,

        "policy_action": action,
        "challenge": challenge,
        "guidance": guidance,

        "warnings": [
            "PAD predictions are experimental and have not been independently validated.",
            "PAD output is not an authentication or access-control decision.",
            "Human detection may miss seated or partially occluded people.",
            "Object labels are general-purpose predictions and may be incorrect.",
            "This prototype does not perform face matching or identity verification.",
        ],
    }
