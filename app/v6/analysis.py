from pathlib import Path
import os
import time

import cv2
import numpy as np

from app.detection import CVDetector
from app.objects import ObjectDetector
from app.quality import assess_quality
from app.vision.captioner import SceneCaptioner
from app.vision.expressions import FacialExpressionAnalyzer
from app.pad.onnx_engine import ONNXPADEngine


MAX_PROCESSING_DIMENSION = 2048


class UnifiedVisionEngine:
    """
    Shared V6 analysis engine for uploaded images and live camera frames.

    Large images are resized for general vision processing.
    Detection coordinates are mapped back to original-image coordinates.
    PAD uses the original decoded image.
    Images are processed in memory and are not persisted.
    PAD output is experimental and is never an authentication decision.
    """

    def __init__(self):
        self.face_detector = CVDetector()
        self.object_detector = ObjectDetector()
        self.captioner = SceneCaptioner()
        self.expression_analyzer = FacialExpressionAnalyzer()

        self.pad_engine = None
        self.pad_error = None

        pad_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "pad"
            / "minifasnet_v2.onnx"
        )

        try:
            self.pad_engine = ONNXPADEngine(str(pad_path))
        except Exception as exc:
            self.pad_error = str(exc)

        self.enable_expressions = (
            os.getenv("ENABLE_EXPRESSION_ANALYZER", "0").lower()
            not in {"0", "false", "no"}
        )

    def status(self):
        return {
            "face_detector": "READY",
            "object_detector": (
                "READY" if self.object_detector.ready else "NOT_READY"
            ),
            "scene_captioner": (
                "READY" if self.captioner.ready
                else "FALLBACK_OR_DISABLED"
            ),
            "expression_analyzer": (
                "ENABLED" if self.enable_expressions else "DISABLED"
            ),
            "pad": (
                "EXPERIMENTAL_NOT_VALIDATED"
                if self.pad_engine is not None
                else "NOT_CONFIGURED"
            ),
            "pad_error": self.pad_error,
            "storage": "DISABLED",
        }

    @staticmethod
    def _prepare_processing_image(image):
        original_height, original_width = image.shape[:2]

        longest_side = max(original_width, original_height)

        if longest_side <= MAX_PROCESSING_DIMENSION:
            return image, 1.0, 1.0

        scale = MAX_PROCESSING_DIMENSION / float(longest_side)

        processing_width = max(1, round(original_width * scale))
        processing_height = max(1, round(original_height * scale))

        processing_image = cv2.resize(
            image,
            (processing_width, processing_height),
            interpolation=cv2.INTER_AREA,
        )

        scale_x = original_width / processing_width
        scale_y = original_height / processing_height

        return processing_image, scale_x, scale_y

    @staticmethod
    def _map_face_box(box, scale_x, scale_y, original_width, original_height):
        x, y, width, height = box

        x1 = max(0, min(original_width, round(float(x) * scale_x)))
        y1 = max(0, min(original_height, round(float(y) * scale_y)))

        x2 = max(
            x1,
            min(original_width, round((float(x) + float(width)) * scale_x)),
        )
        y2 = max(
            y1,
            min(original_height, round((float(y) + float(height)) * scale_y)),
        )

        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def _map_object_box(box, scale_x, scale_y, original_width, original_height):
        x1, y1, x2, y2 = box

        mapped_x1 = max(0, min(original_width, round(float(x1) * scale_x)))
        mapped_y1 = max(0, min(original_height, round(float(y1) * scale_y)))
        mapped_x2 = max(0, min(original_width, round(float(x2) * scale_x)))
        mapped_y2 = max(0, min(original_height, round(float(y2) * scale_y)))

        return [
            mapped_x1,
            mapped_y1,
            max(0, mapped_x2 - mapped_x1),
            max(0, mapped_y2 - mapped_y1),
        ]

    def analyze(self, image: np.ndarray, include_scene: bool = True):
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        started = time.perf_counter()

        original_height, original_width = image.shape[:2]

        processing_image, scale_x, scale_y = (
            self._prepare_processing_image(image)
        )

        # Face detection runs on the processing copy.
        face_result = self.face_detector.detect(processing_image)
        processing_face_boxes = list(face_result.face_boxes)

        # Map face boxes back to original-image coordinates.
        face_boxes = [
            self._map_face_box(
                box,
                scale_x,
                scale_y,
                original_width,
                original_height,
            )
            for box in processing_face_boxes
        ]

        # Object detection runs on the same processing copy.
        objects = self.object_detector.detect(processing_image)

        object_results = []
        person_boxes = []

        for item in objects:
            processing_box = list(map(int, item.box))

            mapped_box = self._map_object_box(
                processing_box,
                scale_x,
                scale_y,
                original_width,
                original_height,
            )

            x1, y1, width, height = mapped_box

            object_results.append({
                "label": item.label,
                "confidence": float(item.confidence),
                "box": [
                    x1,
                    y1,
                    x1 + width,
                    y1 + height,
                ],
            })

            if item.label.lower() == "person":
                person_boxes.append(mapped_box)

        # Preserve the original-image face-width threshold.
        original_face_width = max(
            (box[2] for box in face_boxes),
            default=0,
        )

        # Image quality is measured on the processing copy.
        quality = assess_quality(
            processing_image,
            processing_face_boxes,
            face_width_px=original_face_width,
        )

        # Scene captioning uses the processing copy.
        scene = (
            self.captioner.describe(
                processing_image,
                [item.label for item in objects],
            )
            if include_scene else None
        )

        scene_result = {
            "description": scene.caption,
            "engine": scene.engine,
            "ready": scene.ready,
            "note": scene.note,
        } if scene is not None else {
            "description": None,
            "engine": "SKIPPED_FOR_LIVE_PERFORMANCE",
            "ready": self.captioner.ready,
            "note": "Scene captioning was skipped for this frame.",
        }

        # Optional expression estimates use processing-image coordinates.
        expressions = {
            "status": "disabled",
            "faces": [],
        }

        if self.enable_expressions:
            expressions = self.expression_analyzer.analyze(
                processing_image,
                processing_face_boxes,
            )

        # PAD uses the original image and original face coordinates.
        pad = {
            "status": "NOT_CONFIGURED",
            "score": None,
            "label": "INCONCLUSIVE",
            "probabilities": None,
            "validated": False,
        }

        if self.pad_engine is not None:
            if len(face_boxes) == 0:
                pad["status"] = "NO_FACE"
            elif len(face_boxes) > 1:
                pad["status"] = "MULTIPLE_FACES"
            elif not quality.passed:
                pad["status"] = "QUALITY_REJECTED"
            else:
                result = self.pad_engine.predict(
                    image,
                    face_boxes[0],
                )

                pad.update({
                    "status": (
                        "EXPERIMENTAL_MODEL_OUTPUT"
                        if result.score is not None
                        else "INCONCLUSIVE"
                    ),
                    "score": result.score,
                    "label": result.label,
                    "probabilities": result.probabilities,
                })

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return {
            "status": "ok",
            "frame": {
                "width": original_width,
                "height": original_height,
            },
            "person_count": len(person_boxes),
            "person_boxes": person_boxes,
            "face_count": len(face_boxes),
            "face_boxes": face_boxes,
            "object_count": len(object_results),
            "objects": object_results,
            "quality": quality.model_dump(),
            "scene": scene_result,
            "expressions": expressions,
            "pad": pad,
            "performance": {
                "inference_latency_ms": latency_ms,
                "device": self.object_detector.device,
            },
            "privacy": {
                "image_storage": "disabled",
                "processing": "in_memory",
            },
            "warnings": [
                "Detection results may be inaccurate.",
                "Facial-expression estimates do not establish a person's actual emotions.",
                "PAD outputs are experimental, unvalidated model estimates.",
                "PAD must not be used as an authentication or access-control decision.",
                "This system does not identify people or match identities.",
            ],
        }


vision_engine = UnifiedVisionEngine()
