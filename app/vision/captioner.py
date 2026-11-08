import os
from dataclasses import dataclass
import numpy as np

@dataclass
class CaptionResult:
    caption: str
    engine: str
    ready: bool
    note: str | None = None

class SceneCaptioner:
    """Local BLIP image captioner. Falls back to a transparent label-based summary."""
    def __init__(self):
        self.processor = None
        self.model = None
        self.error = None
        self.device = "cpu"
        self.model_id = os.getenv("SCENE_MODEL", "Salesforce/blip-image-captioning-base")
        self.enabled = os.getenv("ENABLE_SCENE_CAPTIONER", "0").lower() not in {"0", "false", "no"}
        self._torch = None

    def _load_model(self):
        if self.model is not None or not self.enabled:
            return
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration

            self._torch = torch
            requested_device = os.getenv("SCENE_DEVICE", "auto").lower()
            self.device = (
                "cuda" if requested_device == "auto" and torch.cuda.is_available()
                else "cpu" if requested_device == "auto"
                else requested_device
            )
            self.processor = BlipProcessor.from_pretrained(self.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            self.error = str(exc)

    @property
    def ready(self):
        return self.model is not None

    def describe(self, frame: np.ndarray, labels: list[str]) -> CaptionResult:
        self._load_model()
        if self.ready:
            try:
                from PIL import Image
                import cv2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with self._torch.inference_mode():
                    output = self.model.generate(**inputs, max_new_tokens=48, num_beams=3)
                caption = self.processor.decode(output[0], skip_special_tokens=True).strip()
                return CaptionResult(caption=caption, engine="BLIP-local", ready=True)
            except Exception as exc:
                self.error = str(exc)

        unique = list(dict.fromkeys(labels))
        if unique:
            readable = ", ".join(unique[:8])
            caption = f"Detected scene items include: {readable}."
            if "person" in unique:
                caption = "A person is present. " + caption
            return CaptionResult(
                caption=caption,
                engine="local-detection-summary",
                ready=False,
                note="BLIP caption model unavailable; this is a summary of detected labels, not a full scene interpretation."
            )
        return CaptionResult(
            caption="No supported objects were detected in this frame.",
            engine="local-detection-summary",
            ready=False,
            note="The object detector only recognizes its trained label categories."
        )
