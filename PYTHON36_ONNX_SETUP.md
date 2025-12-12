# Python 3.6 + ONNX Runtime Setup Guide

## Overview
The pipeline has been converted to Python 3.6 compatibility using ONNX Runtime instead of ultralytics.

## Changes Made

### 1. Export YOLO to ONNX
- Created `export_yolo_to_onnx.py` for model conversion
- YOLO11n-Pose model exported to ONNX format
- File: `assets/models/yolo11n_hand_pose.onnx` (11.6 MB)

### 2. Replace ultralytics with ONNX Runtime
- Removed `from ultralytics import YOLO`
- Added `import onnxruntime as ort`
- Implemented custom preprocessing, NMS, and postprocessing
- Full YOLO inference pipeline in pure Python 3.6

### 3. Python 3.6 Compatibility
- All f-strings converted to `.format()`
- `super()` calls use explicit class name
- Type hints compatible with Python 3.6 (removed in pipeline.py for simplicity)

## Installation (Jetson Nano - Python 3.6)

### Step 1: Install ONNX Runtime
```bash
# ONNX Runtime for Python 3.6
pip3 install onnxruntime==1.10.0
```

### Step 2: Install other dependencies
```bash
# OpenCV
pip3 install opencv-python

# NumPy
pip3 install numpy

# MediaPipe (for face tracking)
pip3 install mediapipe

# Pathlib (usually included in Python 3.6+)
# If missing: pip3 install pathlib
```

### Step 3: Copy ONNX model
```bash
# The .onnx file is already in assets/models/
# Verify it exists:
ls -lh assets/models/yolo11n_hand_pose.onnx
```

## Usage

### Run on Jetson Nano (Python 3.6)
```bash
# With GStreamer camera
python3.6 src/emoji_reactor/app.py

# With USB camera
python3.6 src/emoji_reactor/app.py --no-gstreamer --camera 0
```

### Run on PC/Mac (for testing)
```bash
# Install ONNX Runtime first
pip3 install onnxruntime

# Run with webcam
python3 src/emoji_reactor/app.py --no-gstreamer --camera 0
```

## Model Details

### ONNX Model Structure
```
Input:  images - [1, 3, 640, 640] (NCHW format, RGB, normalized [0-1])
Output: output0 - [1, 68, 8400]
        68 = 4 (bbox) + 1 (confidence) + 63 (21 keypoints * 3)
        8400 = number of anchor boxes
```

### Processing Pipeline
1. **Preprocess**: Letterbox resize → BGR→RGB → Normalize [0,1] → HWC→CHW → Add batch
2. **Inference**: ONNX Runtime session.run()
3. **Postprocess**: Transpose → Filter by confidence → NMS → Scale back to original image

### Performance Notes
- ONNX Runtime supports CUDA on Jetson Nano (auto-detected)
- If CUDA available: Uses `CUDAExecutionProvider`
- Otherwise: Falls back to `CPUExecutionProvider`

## Exporting New Models

If you need to export pruned or optimized models:

```bash
# Run on system with Python 3.8+ and ultralytics
python3 export_yolo_to_onnx.py
```

This will export:
- `yolo11n_hand_pose.onnx` (base model)
- `yolo11n_hand_pose_pruned_XX.onnx` (if pruned models exist)

## Troubleshooting

### Error: "onnxruntime not installed"
```bash
pip3 install onnxruntime
```

### Error: "ONNX model not found"
```bash
# Re-export from PyTorch model
python3 export_yolo_to_onnx.py
```

### Low FPS on Jetson Nano
- Ensure CUDA is enabled (check log: "[ONNX] CUDA available")
- Consider using TensorRT for better performance:
  ```bash
  pip3 install onnx-tensorrt
  # Convert ONNX → TensorRT engine
  ```

### MediaPipe errors
```bash
# Reinstall MediaPipe
pip3 install --upgrade mediapipe
```

## File Structure
```
Embedded_AI/
├── assets/models/
│   ├── yolo11n_hand_pose.pt          # Original PyTorch model
│   └── yolo11n_hand_pose.onnx        # ONNX export (for Python 3.6)
├── src/
│   ├── hand_tracking/
│   │   ├── __init__.py
│   │   └── pipeline.py               # ONNX Runtime implementation
│   └── emoji_reactor/
│       └── app.py                    # Main application
├── export_yolo_to_onnx.py            # ONNX export script
└── PYTHON36_ONNX_SETUP.md            # This file
```

## Dependencies Summary

| Package | Python 3.6 Compatible | Notes |
|---------|----------------------|-------|
| onnxruntime | ✅ Yes (1.10.0+) | YOLO inference |
| mediapipe | ✅ Yes | Face tracking |
| opencv-python | ✅ Yes | Image processing |
| numpy | ✅ Yes | Numerical operations |
| pathlib | ✅ Yes (built-in) | File paths |
| ultralytics | ❌ No (requires 3.8+) | Not needed (replaced with ONNX) |

## Comparison: ultralytics vs ONNX Runtime

| Feature | ultralytics | ONNX Runtime |
|---------|------------|--------------|
| Python Version | 3.8+ | 3.6+ |
| Installation Size | ~500 MB | ~50 MB |
| Dependencies | Many (torch, etc.) | Minimal |
| Jetson Nano | ⚠️ Limited support | ✅ Full support |
| Inference Speed | Fast | Similar (with CUDA) |
| Model Format | .pt | .onnx |

## Next Steps

1. ✅ Export YOLO to ONNX (done)
2. ✅ Replace ultralytics with ONNX Runtime (done)
3. ✅ Python 3.6 compatibility (done)
4. ⏳ Test on Jetson Nano with Python 3.6
5. ⏳ Optimize with TensorRT (optional, for maximum performance)
