
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import api
from app.main import app


client = TestClient(app)


class FakeDetection:
    def __init__(self, face_boxes):
        self.face_boxes = face_boxes


class FakeQuality:
    def __init__(self, passed):
        self.passed = passed
        self.reasons = (
            [] if passed else ["Test quality failure"]
        )

    def model_dump(self):
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "brightness": 120,
            "blur_score": 100,
            "face_width_px": 100,
        }

class FakePADResult:
    configured = True
    score = 0.80
    label = "MODEL_ESTIMATE_LIVE"
    probabilities = {
        "live": 0.80,
        "print_attack": 0.10,
        "replay_attack": 0.10,
    }


class FakePADEngine:
    def __init__(self):
        self.calls = 0

    def predict(self, image, face_box):
        self.calls += 1
        return FakePADResult()


def make_test_image():
    success, encoded = cv2.imencode(
        ".jpg",
        np.zeros((240, 320, 3), dtype=np.uint8),
    )
    assert success
    return encoded.tobytes()


def run_api_test(monkeypatch, face_boxes, quality_passed):
    engine = FakePADEngine()

    monkeypatch.setattr(
        api,
        "detector",
        type(
            "Detector",
            (),
            {
                "detect": lambda self, image: FakeDetection(face_boxes)
            },
        )(),
    )

    monkeypatch.setattr(
        api,
        "expression_analyzer",
        type(
            "Expression",
            (),
            {
                "analyze": lambda self, image, boxes: {
                    "status": "success",
                    "faces": [],
                    "error": None,
                }
            },
        )(),
    )

    monkeypatch.setattr(
        api,
        "object_detector",
        type(
            "Objects",
            (),
            {
                "ready": False,
                "error": None,
                "detect": lambda self, image: [],
            },
        )(),
    )

    monkeypatch.setattr(
        api,
        "scene_captioner",
        type(
            "Caption",
            (),
            {
                "describe": lambda self, image, labels: type(
                    "Scene",
                    (),
                    {
                        "caption": "Test scene",
                        "engine": "test",
                        "ready": False,
                        "note": None,
                    },
                )()
            },
        )(),
    )

    monkeypatch.setattr(
        api,
        "assess_quality",
        lambda image, boxes: FakeQuality(quality_passed),
    )

    monkeypatch.setattr(api, "pad_engine", engine)

    response = client.post(
        "/api/analyze",
        files={
            "frame": (
                "test.jpg",
                make_test_image(),
                "image/jpeg",
            )
        },
    )

    return response, engine


def test_no_face_skips_pad(monkeypatch):
    response, engine = run_api_test(monkeypatch, [], False)

    assert response.status_code == 200
    assert response.json()["pad_status"] == "NO_FACE"
    assert response.json()["pad_score"] is None
    assert engine.calls == 0


def test_multiple_faces_skip_pad(monkeypatch):
    boxes = [
        (10, 10, 100, 100),
        (150, 10, 100, 100),
    ]

    response, engine = run_api_test(monkeypatch, boxes, False)

    assert response.status_code == 200
    assert response.json()["pad_status"] == "MULTIPLE_FACES"
    assert response.json()["pad_score"] is None
    assert engine.calls == 0


def test_bad_quality_skips_pad(monkeypatch):
    boxes = [(10, 10, 100, 100)]

    response, engine = run_api_test(monkeypatch, boxes, False)

    assert response.status_code == 200
    assert response.json()["pad_status"] == "QUALITY_REJECTED"
    assert response.json()["pad_score"] is None
    assert engine.calls == 0


def test_good_quality_calls_pad(monkeypatch):
    boxes = [(10, 10, 100, 100)]

    response, engine = run_api_test(monkeypatch, boxes, True)

    assert response.status_code == 200
    assert response.json()["pad_status"] == "EXPERIMENTAL_MODEL_OUTPUT"
    assert response.json()["pad_label"] == "MODEL_ESTIMATE_LIVE"
    assert engine.calls == 1


def test_pad_exception_is_handled(monkeypatch):
    boxes = [(10, 10, 100, 100)]

    response, engine = run_api_test(monkeypatch, boxes, True)

    # This test uses a separate failing engine.
    class FailingPADEngine:
        def predict(self, image, face_box):
            raise RuntimeError("Simulated inference failure")

    monkeypatch.setattr(api, "pad_engine", FailingPADEngine())

    response = client.post(
        "/api/analyze",
        files={
            "frame": (
                "test.jpg",
                make_test_image(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["pad_status"] == "INFERENCE_ERROR"
    assert response.json()["pad_label"] == "INCONCLUSIVE"
