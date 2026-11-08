import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load backend environment variables before importing API/auth modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from app.v6.api import router


app = FastAPI(
    title="MOSIP LiveShield V6",
    version="6.0.0",
    description="Unified Vision Intelligence Platform",
)

# Explicit website-origin allowlist.
# Production Cloudflare origin will be added through CORS_ORIGINS.
cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5174,http://127.0.0.1:5174",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(router)

WEB_DIR = PROJECT_ROOT / "web" / "v6"

app.mount(
    "/static",
    StaticFiles(directory=WEB_DIR),
    name="v6-static",
)


@app.get("/")
def index():
    return {
        "project": "MOSIP LiveShield V6",
        "dashboard": "/static/index_v6.html",
        "api_docs": "/docs",
    }