
import numpy as np

from app.detection import CVDetector


def test_yunet_detector_returns_drawable_face_boxes():
    detector = CVDetector()

    class FakeYuNet:
        def setInputSize(self, size):
            self.input_size = size

        def detect(self, frame):
            assert frame.shape[:2] == (160, 120)

            # YuNet format: x, y, width, height, landmarks..., score
            faces = np.array([
                [12, 18, 40, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]
            ], dtype=np.float32)

            return 1, faces

    detector.face_detector = FakeYuNet()

    result = detector.detect(
        np.zeros((160, 120, 3), dtype=np.uint8)
    )

    assert result.face_boxes == [(12, 18, 40, 44)]
    assert result.person_count == 0
    assert result.person_boxes == []


def test_yunet_detector_handles_no_faces():
    detector = CVDetector()

    class FakeYuNet:
        def setInputSize(self, size):
            self.input_size = size

        def detect(self, frame):
            return 1, None

    detector.face_detector = FakeYuNet()

    result = detector.detect(
        np.zeros((160, 120, 3), dtype=np.uint8)
    )

    assert result.face_boxes == []
