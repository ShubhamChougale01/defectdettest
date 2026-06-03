# 🔍 Defect Detection — iOS App
> **On-device structural defect detection using YOLOv8 + CoreML — no server, no cloud, just your iPhone.**
[![Platform](https://img.shields.io/badge/Platform-iOS%2016%2B-black?style=flat-square&logo=apple)](https://developer.apple.com)
[![Model](https://img.shields.io/badge/Model-YOLOv8%20%2B%20CoreML-orange?style=flat-square)](https://ultralytics.com) [![Framework
](https://img.shields.io/badge/UI-SwiftUI-blue?style=flat-square&logo=swift)](https://developer.apple.com/xcode/swiftui)
[![License](https://img.shields.io/badge/License-AGPL--3.0-red?style=flat-square)](https://ultralytics.com/license)
---
## What Is This?
A SwiftUI iOS app that detects real-world structural defects — **cracks, mold, water damage, pipe leaks** — directly on-device
using a YOLOv8 model converted to CoreML. Point your camera or pick a photo, and the model draws bounding boxes with class labels
and confidence scores in real time.
No network calls. No API keys. Everything runs locally on the device.
---
## Features
| Feature | What It Does | Why It Matters |
|---|---|---|
| 🔍 **Real-Time Detection** | YOLOv8 runs inference on every frame via CoreML | Catches defects instantly without latency |
| 📸 **Camera + Gallery Input** | Pick from photo library or capture live | Works in the field or from existing photos |
| 🏷️ **Bounding Boxes + Labels** | Draws boxes with class name and confidence % | Tells you exactly what was found and where |
| 🔁 **Fallback Class Labels** | Uses hardcoded labels if model metadata is missing | Prevents crashes on model version changes |
| 📦 **Fully On-Device** | CoreML inference — no server required | Works offline, no data leaves the device |
| ⚡ **Optimized .mlpackage** | Final model shipped as `.mlpackage` not `.mlmodel` | Faster startup and better memory management |
---
## ⚡ Quick Setup (3 Steps)
### Step 1 — Clone the repo
```bash
git clone https://github.com/ShubhamChougale01/AI-work.git
cd AI-work
Step 2 — Open in Xcode
open defectdettest/defectdettest.xcodeproj
Step 3 — Build & Run
Select a simulator or physical device (iPhone recommended for camera) and press ⌘R.
▎ Note: CoreML inference is significantly faster on a physical device than on a simulator.
---
How It Works
User opens app
      │
      ▼
Select image (camera / photo library)
      │
      ▼
ContentView passes image to CoreML pipeline
      │
      ▼
defect_detect.mlpackage runs YOLOv8 inference
      │
      ├── Bounding box coordinates (x, y, w, h)
      ├── Class label (Crack / Mold / Water Damage / Pipe Leak…)
      └── Confidence score (0.0 – 1.0)
      │
      ▼
BoundingBoxView overlays results on original image
      │
      ▼
User sees annotated image with defect labels
---
Folder Structure
defectdettest/
│
├── defectdettestApp.swift              # App entry point — SwiftUI lifecycle
├── ContentView.swift                   # Core UI: image selection + inference trigger
├── BoundingBoxView.swift               # Draws bounding boxes + labels over image
│
├── defect_detect.pt                    # Original YOLOv8 PyTorch weights (training artifact)
├── DefectDetectionModel.mlmodel        # Converted CoreML model (legacy format)
├── defect_detect.mlpackage/            # Final .mlpackage used at runtime (preferred)
│
├── Assets.xcassets/                    # App icons, colors, image assets
├── Item.swift                          # Data model helpers
│
├── defectdettest.xcodeproj/            # Xcode project config
├── defectdettestTests/                 # Unit tests
└── defectdettestUITests/               # UI automation tests
---
Model Details
┌────────────────┬────────────────────────────────────────────────────┐
│    Property    │                       Value                        │
├────────────────┼────────────────────────────────────────────────────┤
│ Architecture   │ YOLOv8 (Ultralytics)                               │
├────────────────┼────────────────────────────────────────────────────┤
│ Conversion     │ PyTorch → CoreML (.mlpackage)                      │
├────────────────┼────────────────────────────────────────────────────┤
│ Input Size     │ 640 × 640 RGB                                      │
├────────────────┼────────────────────────────────────────────────────┤
│ Output         │ Bounding boxes + class labels + confidence scores  │
├────────────────┼────────────────────────────────────────────────────┤
│ Inference      │ On-device via Apple Neural Engine / GPU            │
├────────────────┼────────────────────────────────────────────────────┤
│ Defect Classes │ Cracks, Water Damage, Mold, Pipe Leak (extendable) │
└────────────────┴────────────────────────────────────────────────────┘
Model files explained:
- defect_detect.pt — raw PyTorch weights. Keep for retraining or re-exporting. Not used at runtime.
- DefectDetectionModel.mlmodel — first-generation CoreML export. Kept for reference.
- defect_detect.mlpackage — the active model. This is what the app loads.
---
Dependencies
- CoreML — on-device model inference (Apple framework, no install needed)
- SwiftUI — declarative UI (Apple framework, no install needed)
- Vision (optional) — for camera preprocessing pipelines
- AVFoundation (optional) — if extending to live video stream detection
No third-party Swift packages required.
---
FAQ
Does this work offline?
Yes. The entire inference pipeline runs on-device using CoreML. No internet connection is needed after installation.
Can I add more defect classes?
Yes — retrain the YOLOv8 model with your new classes, re-export to CoreML using yolo export model=defect_detect.pt format=coreml,
and replace defect_detect.mlpackage.
Why are there two model files (.mlmodel and .mlpackage)?
.mlpackage is the newer, preferred format — faster to load, better hardware utilization. The .mlmodel is kept as a fallback
artifact. The app uses .mlpackage.
Will it work on an iPhone simulator?
It will build and run, but the camera is unavailable on simulators. Use a physical device for real-world testing and to get
accurate inference speed.
What iOS version is required?
iOS 16+ is recommended for full CoreML .mlpackage support.
---
License
The YOLOv8 model is licensed under AGPL-3.0.
See Ultralytics License (https://ultralytics.com/license) for full terms.
The SwiftUI application code in this repository is available under MIT.
