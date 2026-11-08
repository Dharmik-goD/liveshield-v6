from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api import router

app = FastAPI(
    title="MOSIP LiveShield API",
    version="0.1.0",
    description="Offline-first face capture, quality and PAD integration prototype.",
)
app.include_router(router)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "service": "mosip-liveshield", "biometric_storage": "disabled"}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")
