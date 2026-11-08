import cv2
import numpy as np
import pytest

from app.pad.onnx_engine import ONNXPADEngine


@pytest.fixture
def engine():
    instance = ONNXPADEngine.__new__(ONNXPADEngine)
    return instance


def test_invalid_image_returns_inconclusive(engine):
    result = engine.predict(None, (10, 10, 50, 50))

    assert result.configured is True
    assert result.label == "INCONCLUSIVE"
    assert result.score is None


def test_invalid_face_box_returns_inconclusive(engine):
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    result = engine.predict(image, (10, 10, -50, 50))

    assert result.label == "INCONCLUSIVE"
    assert result.score is None


def test_non_finite_face_coordinates_return_inconclusive(engine):
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    result = engine.predict(
        image,
        (float("nan"), 10, 50, 50),
    )

    assert result.label == "INCONCLUSIVE"
    assert result.score is None


def test_unexpected_runtime_error_returns_inconclusive(engine):
    class BrokenSession:
        def run(self, outputs, inputs):
            raise RuntimeError("Simulated ONNX failure")

    engine.session = BrokenSession()
    engine.input_name = "input"
    engine.output_name = "output"

    image = np.zeros((240, 320, 3), dtype=np.uint8)

    result = engine.predict(image, (50, 50, 80, 80))

    assert result.label == "INCONCLUSIVE"
    assert result.score is None


def test_nan_model_output_returns_inconclusive(engine):
    class NaNSession:
        def run(self, outputs, inputs):
            return [np.array([[np.nan, 0.1, 0.2]], dtype=np.float32)]

    engine.session = NaNSession()
    engine.input_name = "input"
    engine.output_name = "output"

    image = np.zeros((240, 320, 3), dtype=np.uint8)

    result = engine.predict(image, (50, 50, 80, 80))

    assert result.label == "INCONCLUSIVE"
    assert result.score is None


def test_infinite_model_output_returns_inconclusive(engine):
    class InfiniteSession:
        def run(self, outputs, inputs):
            return [np.array([[np.inf, 0.1, 0.2]], dtype=np.float32)]

    engine.session = InfiniteSession()
    engine.input_name = "input"
    engine.output_name = "output"

    image = np.zeros((240, 320, 3), dtype=np.uint8)

    result = engine.predict(image, (50, 50, 80, 80))

    assert result.label == "INCONCLUSIVE"
    assert result.score is None
def test_crop_matches_upstream_for_centered_face(engine):
    image = np.arange(240 * 320 * 3, dtype=np.uint8).reshape(240, 320, 3)
    face_box = (100, 70, 60, 60)

    actual = engine._prepare_face(image, face_box)

    x1, y1, x2, y2 = engine._get_new_box(
        320, 240, face_box, 2.7
    )
    expected_crop = image[y1:y2 + 1, x1:x2 + 1]
    expected_resized = cv2.resize(expected_crop, (80, 80))
    expected = np.transpose(
        expected_resized.astype(np.float32) / 255.0,
        (2, 0, 1),
    )[None, ...]

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (1, 3, 80, 80)
    assert actual.dtype == np.float32


def test_crop_matches_upstream_near_image_boundary(engine):
    image = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
    face_box = (2, 3, 45, 45)

    actual = engine._prepare_face(image, face_box)

    x1, y1, x2, y2 = engine._get_new_box(
        160, 120, face_box, 2.7
    )
    expected_crop = image[y1:y2 + 1, x1:x2 + 1]
    expected_resized = cv2.resize(expected_crop, (80, 80))
    expected = np.transpose(
        expected_resized.astype(np.float32) / 255.0,
        (2, 0, 1),
    )[None, ...]

    np.testing.assert_allclose(actual, expected)
