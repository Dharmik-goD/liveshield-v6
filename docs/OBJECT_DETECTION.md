# General object recognition

LiveShield V2 adds YOLOv8n object detection for common objects from the COCO
label set, such as person, bicycle, car, bottle, chair, laptop, mobile phone,
book and cup. The API returns labels, confidence scores and pixel bounding boxes.

## Setup
Install with `pip install -r requirements.txt`. On first startup, Ultralytics
may download `yolov8n.pt`, requiring internet once. Inference then runs locally
on CPU. Set `OBJECT_MODEL` to another compatible model path if required.

## Limitations
This is object detection, not unrestricted visual understanding. It only
recognizes classes in its trained label set; unknown objects may be missed or
mislabelled. Confidence is not a calibrated probability. Object detection is
separate from biometric PAD and cannot determine whether a face is live or spoofed.
Review Ultralytics/model licensing before competition or commercial distribution.
