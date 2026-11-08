import builtins
import numpy as np

from app import api as api_module
from app.main import app, health
from app.objects import ObjectDetector
from app.vision.captioner import SceneCaptioner
from app.vision.expressions import FacialExpressionAnalyzer


def test_app_import_preserves_routes_and_reports_experimental_pad():
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    paths.update(route.path for route in api_module.router.routes)

    assert {"/health", "/api/info", "/api/analyze", "/"}.issubset(paths)
    assert health()["status"] == "ok"

    info = api_module.info()
    assert info["pad_model_configured"] is True
    assert info["pad_status"] == "EXPERIMENTAL_NOT_VALIDATED"
    assert info["pad_model_error"] is None


def test_optional_vision_services_fail_soft_without_ai_packages(monkeypatch):
    monkeypatch.delenv("ENABLE_SCENE_CAPTIONER", raising=False)
    original_import = builtins.__import__

    def import_without_optional_ai(name, *args, **kwargs):
        if name == "torch" or name.startswith(("ultralytics", "transformers")):
            raise ImportError(f"simulated missing optional package: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_optional_ai)

    detector = ObjectDetector()
    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []

    expression = FacialExpressionAnalyzer().analyze(
        np.zeros((64, 64, 3), dtype=np.uint8), [(0, 0, 32, 32)]
    )
    assert expression["status"] == "unavailable"
    assert expression["faces"] == []

    caption = SceneCaptioner().describe(
        np.zeros((64, 64, 3), dtype=np.uint8), []
    )
    assert caption.ready is False
    assert caption.engine == "local-detection-summary"
