# YOLO + QML Survivor Priority Project

## What the project does

This project detects humans in a live mobile camera/video feed and classifies each detected person as a Low, Medium, or High priority survivor.

Pipeline:

```text
Video / Mobile Camera
    -> YOLOv8 human detection
    -> Fire/smoke YOLO hazard detection
    -> Survivor feature extraction
    -> QSVM priority classifier
    -> Low / Medium / High survivor priority
```

## Models used

- `yolov8m.pt`: detects people.
- `runs/detect/runs/fire_smoke/yolov8m_fire_smoke/weights/best.pt`: trained on the provided fire/smoke YOLOv8 dataset.
- `priority_qsvm.pkl`: QSVM model for survivor priority.
- `priority_scaler.pkl`: scaler used before QSVM prediction.

## Fire/smoke dataset

Dataset file:

```text
fire and smoke detection.v1i.yolov8.zip
```

Extracted to:

```text
datasets/fire_smoke_yolov8
```

Classes:

```text
fire
smoke
```

Split:

```text
train: 333 images
valid: 95 images
test: 48 images
```

## Current validation result

The fire/smoke model was trained for a short CPU-friendly 3-epoch run.

Validation result:

```text
mAP50: 0.149
mAP50-95: 0.050
fire recall: 0.75
smoke recall: 0.235
```

This is enough to demonstrate the integrated pipeline, but it is not yet a production-quality hazard detector. More epochs, GPU training, and more diverse data would improve it.

## QSVM features

The QSVM receives:

```text
distance_from_camera
body_visibility_pct
movement_detected
nearby_hazards
time_since_last_movement
```

The important improvement is that `nearby_hazards` now comes from trained fire/smoke YOLO detections when the trained model is available. The system is conservative by default: fire/smoke must pass a high confidence threshold and be near the detected person before it affects survivor priority.

HSV color fallback is disabled by default because it can create false positives in normal scenes. It can be enabled only for debugging with:

```powershell
python survivor_detection.py --enable-color-fallback
```

## Commands

Generate survivor-priority dataset:

```powershell
python generate_priority_dataset.py
```

Train QSVM priority model:

```powershell
python train_qsvm_priority.py
```

Train fire/smoke YOLO model:

```powershell
python train_fire_smoke_yolo.py
```

Run on mobile camera:

```powershell
python survivor_detection.py --source mobile --mobile-ip 192.168.1.103:8080
```

Run on video without display window:

```powershell
python survivor_detection.py --source survivor_detection_output.mp4 --output survivor_review_demo.mp4 --no-window
```

Short test run:

```powershell
python survivor_detection.py --source survivor_detection_output.mp4 --output survivor_review_demo.mp4 --no-window --max-frames 10
```

Show separate fire/smoke debug boxes:

```powershell
python survivor_detection.py --source survivor_detection_output.mp4 --show-hazard-boxes
```

Run with GPS + survivor report output:

```powershell
python gps_server.py
python survivor_detection.py --source mobile --mobile-ip 192.168.1.103:8080
```

After the detection run, outputs are generated in `reports`:

```text
reports/survivor_report.html
reports/survivor_map.html
reports/survivors.json
reports/survivors.csv
reports/images/
```

The report shows each captured survivor image, the QML/QSVM priority, GPS coordinates, movement state, nearby hazard, and a map link. The map plots survivor GPS markers by priority.

## Best explanation for review

This project combines classical computer vision and quantum machine learning. YOLO detects humans and hazards like fire or smoke. The extracted rescue-relevant features are passed to a QSVM, which classifies each detected survivor into Low, Medium, or High priority. The project demonstrates how QML can be used as a decision layer on top of real-time visual detections.

## Honest limitation

The system is a prototype. Human detection is strong because it uses pretrained YOLOv8, but the fire/smoke detector was trained quickly on CPU for demonstration. The QSVM priority model also uses synthetic priority labels. For real deployment, the next step is collecting real rescue-scene data with expert priority annotations.
