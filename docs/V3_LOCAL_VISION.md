# LiveShield V3 — local object recognition + scene description

## Two independent vision outputs
1. YOLO object detector: returns supported class labels, confidence scores and boxes.
2. BLIP scene captioner: generates a natural-language caption from the whole image,
   running locally after model files have been downloaded/cached.

Human/face detection and the PAD adapter remain separate modules. Object recognition
or a scene caption must never be used as a PAD/liveness result or identity decision.

## Baseline installation
```powershell
pip install -r requirements.txt
```

## Optional natural-language scene captioning
Install a compatible PyTorch build first. For NVIDIA GPU, use the official PyTorch
selector at https://pytorch.org/get-started/locally/ to choose the CUDA wheel that
matches the installed driver. If uncertain, install the CPU build. Then:
```powershell
pip install -r requirements-vision.txt
```

The first run downloads the BLIP processor/model and caches it locally; internet is
required for this one-time download unless you pre-populate the Hugging Face cache.
Subsequent inference can run offline. Default device selection uses CUDA when PyTorch
reports it available, otherwise CPU. Set `SCENE_DEVICE=cpu` to force CPU.

## What happens if BLIP is unavailable?
The app still works. It produces a clearly labelled summary from YOLO detections,
e.g. "A person is present. Detected scene items include: person, laptop, chair."
The API says `scene_model_ready: false` so this is not misrepresented as a generated
image caption.

## API additions
`POST /api/analyze` returns:
- `objects`: label, confidence and pixel bounding box
- `scene_description`: natural-language caption or fallback label summary
- `scene_description_engine`: `BLIP-local` or `local-detection-summary`
- `scene_model_ready`: whether the BLIP model actually loaded
- `scene_description_note`: fallback/limitation note

## Limitations
Captions can be wrong or omit details. Object models only recognize trained classes.
This system does not provide unrestricted visual question answering. Do not infer
sensitive attributes, identity, intent, or PAD status from captions.
