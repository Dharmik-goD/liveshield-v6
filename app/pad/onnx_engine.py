from pathlib import Path

import cv2
import numpy as np

from app.pad.base import PADEngine, PADResult


class ONNXPADEngine(PADEngine):
    """
    Experimental MiniFASNet V2 ONNX adapter.

    Input: BGR, NCHW, float32, 80x80, values in [0, 1].
    Output: live, print-attack, replay-attack logits.
    """

    CLASS_NAMES = ("live", "print_attack", "replay_attack")
    CROP_SCALE = 2.7

    def __init__(self, model_path: str):
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        self.session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("Unexpected ONNX input/output count")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        if inputs[0].shape[1:] != [3, 80, 80]:
            raise RuntimeError(
                f"Unexpected model input shape: {inputs[0].shape}"
            )

        if outputs[0].shape[1:] != [3]:
            raise RuntimeError(
                f"Unexpected model output shape: {outputs[0].shape}"
            )

    def _prepare_face(self, image: np.ndarray, face_box: tuple) -> np.ndarray:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image")

        x, y, width, height = map(float, face_box)

        if width <= 0 or height <= 0:
            raise ValueError("Invalid face bounding box")

        image_height, image_width = image.shape[:2]

        center_x = x + width / 2.0
        center_y = y + height / 2.0

        crop_width = width * self.CROP_SCALE
        crop_height = height * self.CROP_SCALE

        x1 = max(0, int(round(center_x - crop_width / 2.0)))
        y1 = max(0, int(round(center_y - crop_height / 2.0)))
        x2 = min(image_width, int(round(center_x + crop_width / 2.0)))
        y2 = min(image_height, int(round(center_y + crop_height / 2.0)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Face crop falls outside the image")

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            raise ValueError("Empty face crop")

        resized = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        return np.ascontiguousarray(tensor, dtype=np.float32)

    def predict(self, image: np.ndarray, face_box: tuple) -> PADResult:
        try:
            tensor = self._prepare_face(image, face_box)

            output = self.session.run(
                [self.output_name],
                {self.input_name: tensor},
            )[0]

            logits = np.asarray(output, dtype=np.float64)

            if logits.shape != (1, 3) or not np.isfinite(logits).all():
                raise ValueError("Invalid model output")

            logits = logits[0]
            shifted = logits - np.max(logits)
            exponentials = np.exp(shifted)
            probabilities = exponentials / np.sum(exponentials)

            probability_map = {
                name: float(probability)
                for name, probability in zip(
                    self.CLASS_NAMES,
                    probabilities,
                )
            }

            predicted_class = int(np.argmax(probabilities))

            labels = (
                "MODEL_ESTIMATE_LIVE",
                "MODEL_ESTIMATE_PRINT_ATTACK",
                "MODEL_ESTIMATE_REPLAY_ATTACK",
            )

            return PADResult(
                configured=True,
                score=float(probabilities[0]),
                label=labels[predicted_class],
                probabilities=probability_map,
            )

        except Exception:
             return PADResult(
        configured=True,
        score=None,
        label="INCONCLUSIVE",
        probabilities=None,
             )
