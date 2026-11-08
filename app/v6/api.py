
import time

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.v6.auth import require_authenticated_user
from app.v6.compute import ComputeManager
from app.v6.tracking import ObjectTracker
from app.v6.analysis import vision_engine


router = APIRouter(prefix="/api/v6", tags=["LiveShield V6"])

compute_manager = ComputeManager()

# Share one YOLO model across all V6 vision features.
object_detector = vision_engine.object_detector

object_tracker = ObjectTracker()



async def decode_upload(frame: UploadFile):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    MAX_UPLOAD_BYTES = 50_000_000
    CHUNK_SIZE = 1_048_576

    if frame.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG or WebP image.",
        )

    # Read incrementally to enforce the upload limit.
    raw = bytearray()

    while True:
        chunk = await frame.read(CHUNK_SIZE)

        if not chunk:
            break

        if len(raw) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 50 MB upload limit.",
            )

        raw.extend(chunk)

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Empty image.",
        )

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image.",
        )

    height, width = image.shape[:2]

    if width * height > 100_000_000:
        raise HTTPException(
            status_code=413,
            detail="Image resolution exceeds the 100-megapixel limit.",
        )

    return image



# PUBLIC: used by deployment health monitoring.
@router.get("/health")
def v6_health():
    return {
        "status": "ok",
        "project": "MOSIP LiveShield V6",
        "version": "6.0.0",
        "storage": "disabled",
    }


# PRIVATE: authenticated team members only.
@router.get("/system")
def v6_system(user=Depends(require_authenticated_user)):
    return {
        "project": "MOSIP LiveShield V6",
        "compute": compute_manager.status(),
        "engines": vision_engine.status(),
        "model_execution": {
            "yolo": object_detector.device,
            "yunet": "OpenCV execution; device not confirmed",
            "pad": "CPUExecutionProvider",
            "scene_captioner": vision_engine.status()["scene_captioner"],
        },
        "privacy": {
            "biometric_storage": "disabled",
            "face_recognition": "not implemented",
            "identity_matching": "not implemented",
        },
    }


@router.get("/vision/status")
def vision_status(user=Depends(require_authenticated_user)):
    object_detector._load_model()

    return {
        "engine": "YOLO",
        "model_ready": object_detector.ready,
        "model_error": object_detector.error,
        "device": object_detector.device,
        "storage": "disabled",
    }


@router.get("/analysis/status")
def analysis_status(user=Depends(require_authenticated_user)):
    object_detector._load_model()

    return {
        "engine": "Unified Vision Intelligence",
        "components": vision_engine.status(),
        "storage": "disabled",
    }


@router.post("/vision/frame")
async def process_vision_frame(
    frame: UploadFile = File(...),
    user=Depends(require_authenticated_user),
):
    image = await decode_upload(frame)
    height, width = image.shape[:2]

    started = time.perf_counter()

    try:
        detections = object_detector.detect(image)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Vision inference failed.",
        ) from exc

    latency_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )

    return {
        "status": "ok",
        "engine": "YOLO",
        "model_ready": object_detector.ready,
        "model_error": object_detector.error,
        "device": object_detector.device,
        "frame": {
            "width": width,
            "height": height,
        },
        "object_count": len(detections),
        "inference_latency_ms": latency_ms,
        "detections": [
            {
                "label": item.label,
                "confidence": item.confidence,
                "box": item.box,
            }
            for item in detections
        ],
        "storage": "disabled",
        "note": "Object detections are model predictions and may be inaccurate.",
    }


@router.post("/analysis")
async def unified_image_analysis(
    frame: UploadFile = File(...),
    include_scene: bool = True,
    user=Depends(require_authenticated_user),
):
    image = await decode_upload(frame)

    try:
        return vision_engine.analyze(
            image,
            include_scene=include_scene,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Unified vision analysis failed.",
        ) from exc


@router.post("/tracking/frame")
async def process_tracking_frame(
    frame: UploadFile = File(...),
    user=Depends(require_authenticated_user),
):
    image = await decode_upload(frame)
    height, width = image.shape[:2]

    started = time.perf_counter()

    try:
        detections = object_detector.detect(image)

        detection_data = [
            {
                "label": item.label,
                "confidence": item.confidence,
                "box": [int(value) for value in item.box],
            }
            for item in detections
        ]

        tracking_result = object_tracker.update(detection_data)

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Tracking inference failed.",
        ) from exc

    faces = []
    face_detection_error = None

    try:
        face_result = vision_engine.face_detector.detect(image)

        for x, y, box_width, box_height in face_result.face_boxes:
            center_x = x + box_width // 2
            center_y = y + box_height // 2

            faces.append({
                "box": [x, y, box_width, box_height],
                "center": [center_x, center_y],
            })

    except Exception:
        face_detection_error = "Face detection unavailable."

    latency_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )

    return {
        "status": "ok",
        "engine": "YOLO + Centroid Tracker + YuNet",
        "device": object_detector.device,
        "frame": {
            "width": width,
            "height": height,
        },
        "inference_latency_ms": latency_ms,
        "faces": faces,
        "face_detection_error": face_detection_error,
        **tracking_result,
        "note": "Tracking IDs are temporary session IDs, not personal identities.",
    }


@router.post("/tracking/reset")
def reset_tracking(user=Depends(require_authenticated_user)):
    object_tracker.reset()

    return {
        "status": "ok",
        "message": "Tracking session reset.",
        "active_tracks": 0,
    }
