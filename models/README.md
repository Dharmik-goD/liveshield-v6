# PAD Model Directory

## Integrated Model

The project includes an experimental ONNX-based presentation attack detection (PAD) model.

- **Model:** MiniFASNet V2
- **Source:** `garciafido/minifasnet-v2-anti-spoofing-onnx`
- **Pinned revision:** `d29c87568ca9b5662da803b10f217c4db20b142b`
- **Model file:** `models/pad/minifasnet_v2.onnx`
- **Runtime:** ONNX Runtime
- **Execution provider:** CPU
- **License declared by source:** Apache License 2.0

## Model Integrity

Expected SHA256:

```text
D7B3CD9BA8A7CEB13BAA8C4720902E27CA3112EFF52F926C08804AF6B6EECC7B
```

Verify the local file using PowerShell:

```powershell
Get-FileHash .\models\pad\minifasnet_v2.onnx -Algorithm SHA256
```

## Input Preprocessing

The adapter implements the model-specific preprocessing:

1. Uses the detected face bounding box.
2. Creates an expanded crop around the face.
3. Resizes the crop to 80 × 80 pixels.
4. Uses BGR image ordering.
5. Converts pixel values to float32 in the [0, 1] range.
6. Converts the image to NCHW format.

Expected input tensor:

```text
[batch, 3, 80, 80]
```

Expected output tensor:

```text
[batch, 3]
```

The three output classes are interpreted as live, print attack, and replay attack. Softmax is applied to the model outputs.

## API Integration

The PAD engine is called only when:

- Exactly one face is detected.
- The image passes the existing quality checks.
- The PAD engine is configured.

The API reports explicit statuses for cases such as:

- `NO_FACE`
- `MULTIPLE_FACES`
- `QUALITY_REJECTED`
- `EXPERIMENTAL_MODEL_OUTPUT`
- `INCONCLUSIVE`
- `INFERENCE_ERROR`

## Validation Status and Limitations

The model has been integrated into the application and its software interface has been tested.

The project test suite currently reports:

```text
12 passed
```

These tests verify application behavior and mocked PAD control flow. They do **not** establish model accuracy, attack detection performance, calibrated confidence, or production readiness.

The model has not yet been independently evaluated or calibrated for this project's intended deployment environment. Its outputs must be treated as experimental model estimates, not definitive LIVE/SPOOF authentication decisions.

Before production use, the model requires independent evaluation using appropriate, consented datasets, documented testing protocols, performance analysis, and threshold calibration.

## Model Files

The ONNX model is stored at:

```text
models/pad/minifasnet_v2.onnx
```

Do not replace the model without updating its source, revision, license information, preprocessing details, output mapping, and SHA256 checksum.