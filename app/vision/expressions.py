import cv2


class FacialExpressionAnalyzer:

    def __init__(self):
        self.model_name = "trpakov/vit-face-expression"
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.error = None
        self._torch = None
        self._image_type = None

    def load_model(self):
        """Load the expression model only when first needed."""

        if self.model is not None:
            return True

        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self._torch = torch
            self._image_type = Image
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name
            )

            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name
            )

            self.model.to(self.device)
            self.model.eval()

            return True

        except Exception as e:
            self.error = str(e)
            return False

    def analyze(self, frame, face_boxes):
        """
        Analyze visible facial-expression patterns.

        frame: OpenCV BGR image
        face_boxes: list of (x, y, width, height)
        """

        if not self.load_model():
            return {
                "status": "unavailable",
                "error": self.error,
                "faces": []
            }

        results = []

        if frame is None:
            return {
                "status": "error",
                "faces": []
            }

        height, width = frame.shape[:2]

        for index, box in enumerate(face_boxes):

            try:
                x, y, w, h = map(int, box)

                # Keep face coordinates inside the image.
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + w)
                y2 = min(height, y + h)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                rgb_face = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )

                pil_face = self._image_type.fromarray(rgb_face)

                inputs = self.processor(
                    images=pil_face,
                    return_tensors="pt"
                )

                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                with self._torch.inference_mode():
                    outputs = self.model(**inputs)
                    probabilities = self._torch.softmax(
                        outputs.logits,
                        dim=-1
                    )[0]

                predicted_index = int(
                    self._torch.argmax(probabilities).item()
                )

                label = self.model.config.id2label[
                    predicted_index
                ]

                confidence = float(
                    probabilities[predicted_index].item()
                )

                results.append({
                    "face_id": index + 1,
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "expression_estimate": str(label),
                    "confidence": round(confidence * 100, 2)
                })

            except Exception as e:
                results.append({
                    "face_id": index + 1,
                    "error": str(e)
                })

        return {
            "status": "success",
            "device": self.device,
            "faces": results,
            "disclaimer": (
                "These are estimates of visible facial expressions, "
                "not a measurement of a person's actual feelings."
            )
        }
