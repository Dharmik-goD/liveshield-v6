# MOSIP LiveShield V3 — BioSentinels

**Offline-first face capture, human detection, quality assessment and PAD integration scaffold** for the MOSIP Decode 2026 problem statement: *Face Liveness Detection and Presentation Attack Detection for Registration and Biometric Authentication*.

> Important: this repository is a runnable engineering MVP/scaffold, not a certified biometric PAD system. The included CV baseline detects people/faces and evaluates capture quality. It deliberately returns **INCONCLUSIVE** for PAD until a trained, validated PAD model is installed. Do not use it to approve real identity enrolments or authentication.

## What is implemented

- Browser camera capture with live preview.
- YOLOv8n general object detection with object names, confidence scores and bounding boxes.
- Optional local BLIP scene captioning that describes the whole camera frame in natural language.
- Human/person detection using OpenCV's built-in HOG people detector.
- Face detection using OpenCV Haar cascade.
- Capture-quality gate (face count, face size, brightness, blur).
- Temporal frame signals (frame-to-frame change) for diagnostics only.
- Policy engine: quality failure → retry; uncertain PAD → active challenge prompt; no silent acceptance.
- API-first FastAPI backend, local/offline inference path, health endpoint.
- Optional ONNX PAD model adapter contract (model is not bundled).
- Privacy-first defaults: frames are processed in memory; no image storage or face templates.
- Tests for policy and quality logic.

## MOSIP problem-statement alignment

The challenge asks Desktop and Android Registration Clients to process face-image streams from biometric devices, perform face liveness/PAD, and guide residents, operators and supervisors through active facial challenges when required. LiveShield provides a modular prototype boundary for those capabilities:

| Challenge need | Repository module |
|---|---|
| Face image stream | `web/index.html`, `app/api.py` |
| Human and face presence | `app/detection.py` |
| Capture quality | `app/quality.py` |
| Passive PAD integration point | `app/pad/onnx_engine.py` |
| Active challenge policy | `app/challenges.py`, `app/policy.py` |
| Desktop/API integration | FastAPI endpoints |
| Offline operation | Local browser + local Python process |
| Privacy | No persistence by default |

The web demo is a desktop prototype. Android integration should consume the same versioned API or package the inference pipeline in an Android-compatible runtime (for example, ONNX Runtime Mobile) after model conversion and device testing.

## Architecture

```text
Browser / biometric-device adapter
          |
          v
      POST /api/analyze  (JPEG frame; local network only)
          |
          v
  Decode + input validation
          |
          +--> Person detector (HOG)
          +--> Face detector (Haar)
          +--> Quality gate (size / blur / exposure / face count)
          +--> Passive PAD adapter (ONNX model, if installed)
          |
          v
       Policy engine
          |
          +--> RETRY_CAPTURE
          +--> ACTIVE_CHALLENGE_REQUIRED
          +--> PAD_INCONCLUSIVE
          +--> PAD_REVIEW_REQUIRED
```

## Quick start (Windows PowerShell)

Use Python **3.11** (the declared supported range is 3.11–3.12).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000`.

The core install deliberately does not require PyTorch, Transformers or Ultralytics.
The API and existing endpoints start without them. Object detection and expression
analysis report unavailable until their optional packages are installed. The bundled
`yolov8n.pt` file is used only as a local model; weights are never downloaded by the app.

For optional vision features, first install a PyTorch build from the official
[PyTorch install selector](https://pytorch.org/get-started/locally/), then install
the other optional packages:

```powershell
pip install ultralytics transformers pillow
```

Scene captions are disabled by default to avoid model downloads on startup. Set
`ENABLE_SCENE_CAPTIONER=1` to opt in; the model loads on the first analysis request.
The expression model also loads only when a frame is analyzed.

The GeForce RTX 5050 is an NVIDIA Blackwell GPU with compute capability 12.0.
Blackwell support requires a compatible CUDA-enabled PyTorch build (CUDA 12.8 or
newer); select a current Windows build from the official PyTorch selector and verify
it on this machine before relying on GPU inference. The application selects CUDA
when PyTorch reports it available, otherwise it uses CPU. `OBJECT_DEVICE=cpu` can
force CPU object detection.

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## API

- `GET /health` — health and model readiness.
- `GET /api/info` — component and safety status.
- `POST /api/analyze` — multipart upload field `frame`; returns detected object names and confidence scores when the YOLO model is ready.

Example:
```powershell
curl.exe -X POST http://127.0.0.1:8000/api/analyze -F "frame=@sample.jpg"
```

The response includes `person_count`, `face_count`, quality metrics, PAD status, policy action and next-step guidance.

## PAD status

PAD is deliberately `NOT_CONFIGURED` in this version. The placeholder ONNX adapter
is not activated by `PAD_MODEL_PATH`, and the API never treats an arbitrary ONNX
model as liveness evidence. A future integration must select a licensed model,
implement its documented preprocessing and output mapping, calibrate thresholds on
held-out data, and report APCER/BPCER, ACER, confidence intervals and condition-level
results. A single still image cannot reliably prove liveness.

## Active challenge design

The prototype provides a challenge prompt/policy boundary. A real implementation should use randomized, short-lived challenges (e.g. blink or a prompted head movement), verify the action from a sequence of frames, expire the challenge, rate-limit retries, and ensure challenge generation cannot be predicted in advance. Do not treat a browser prompt alone as proof of liveness.

## Evaluation plan

Create a subject-disjoint and session-disjoint test split. Include varied cameras, lighting, distances and skin tones. Record:

- APCER per presentation-attack species.
- BPCER for bona fide presentations.
- ACER (reported alongside APCER/BPCER, never alone).
- Detection/decision latency on target desktop and Android devices.
- Failure-to-acquire and retry rates.
- Results by device, illumination and demographic test group.

Use ISO/IEC 30107-1 terminology/framework and the applicable PAD testing guidance. This code does **not** claim ISO certification or conformance.

## Security and privacy notes

- Bind to `127.0.0.1` for local development. Do not expose this development server publicly.
- Add authenticated device registration, TLS, signed model manifests, secure update and tamper-evident audit controls before integration.
- Never log raw frames, face crops, embeddings or biometric templates.
- Keep audit records minimal: model/version, decision category, quality outcome, timestamps and non-identifying transaction IDs.
- Add operator/supervisor override controls only with explicit policy and audit.
- A PAD result is not face matching or identity verification. Keep those as separate stages.

## Repository map

```text
app/
  main.py                 FastAPI app and routes
  api.py                  Frame request handling
  detection.py            Person + face detection
  quality.py              Capture quality gate
  temporal.py             Frame sequence diagnostics
  challenges.py           Active challenge prompts
  policy.py               Decision orchestration
  schemas.py              Typed API response
  pad/
    base.py               PAD adapter interface
    onnx_engine.py        Optional ONNX Runtime adapter
web/
  index.html              Browser camera demo
tests/
  test_policy.py
  test_quality.py
models/
  README.md               Model installation contract
docs/
  MOSIP_ALIGNMENT.md
  EVALUATION_PROTOCOL.md
```

## License

This scaffold is released under the MIT License. Third-party libraries and any future model weights retain their own licences. Review all dataset/model licences before redistribution.

## V3 local scene understanding

See `docs/V3_LOCAL_VISION.md` for optional BLIP setup, offline caching and limitations. Human detection, object detection and PAD remain separate decision paths.
