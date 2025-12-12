# Python 3.6 Compatibility Changes

## Summary
Converted entire emoji reactor pipeline to Python 3.6 compatibility.

## Changes Made

### 1. Removed GStreamer Display Support
- Removed `--gst-display` argument
- Removed nvoverlaysink hardware-accelerated display code
- Removed keyboard input handling (termios/select) for GStreamer mode
- Simplified to cv2.imshow only

### 2. Converted All F-Strings to .format()

#### app.py
- Line 56: `f"[Warning] Could not load {path}"` → `"[Warning] Could not load {}".format(path)`
- Line 118: `f"[BackgroundMusic] File not found: {self.path}"` → `"[BackgroundMusic] File not found: {}".format(self.path)`
- Line 154: `f"[BackgroundMusic] Error: {e}"` → `"[BackgroundMusic] Error: {}".format(e)`
- Line 170: `f"{sound_name}.wav"` → `"{}.wav".format(sound_name)`
- Line 174: `f"{sound_name}.mp3"` → `"{}.mp3".format(sound_name)`
- Line 213: `f"[Audio] Background music: {bg_music_path.name} on {args.audio_device}"` → `"[Audio] Background music: {} on {}".format(...)`
- Line 226: `f"Opening camera {args.camera}..."` → `"Opening camera {}...".format(args.camera)`
- Line 288: `f"{state} {emoji_char}"` → `"{} {}".format(state, emoji_char)`
- Line 289: `f"FPS {np.mean(fps_hist):.0f}"` → `"FPS {:.0f}".format(np.mean(fps_hist))`

#### pipeline.py
- Line 61: `f"{precision}"` → `"{}".format(precision)`
- Line 63: `f", pruned {int(prune_rate*100)}%"` → `", pruned {}%".format(int(prune_rate*100))`
- Line 65: `f"[Pipeline] Loading YOLO11n-Pose hand model ({model_desc})..."` → `"[Pipeline] Loading YOLO11n-Pose hand model ({})...".format(model_desc)`
- Line 71: `f"assets/models/yolo11n_hand_pose_pruned_{prune_pct}.pt"` → `"assets/models/yolo11n_hand_pose_pruned_{}.pt".format(prune_pct)`
- Line 73: `f"[Warning] Pruned model not found, using base model"` → `"[Warning] Pruned model not found, using base model"`
- Line 78: `f"[Warning] INT8 TensorRT engine not found, using FP32"` → `"[Warning] INT8 TensorRT engine not found, using FP32"`
- Line 83: `f"[Warning] FP16 TensorRT engine not found, using FP32"` → `"[Warning] FP16 TensorRT engine not found, using FP32"`
- Line 89: `f"Model not found: {model_path}"` → `"Model not found: {}".format(model_path)`
- Line 92: `f"[Pipeline] YOLO11n-Pose loaded (21 keypoints per hand, {model_desc})"` → `"[Pipeline] YOLO11n-Pose loaded (21 keypoints per hand, {})".format(model_desc)`
- Line 142: `f"[Hand] Error: {e}"` → `"[Hand] Error: {}".format(e)`
- Line 174: `f"[Face] Error: {e}"` → `"[Face] Error: {}".format(e)`
- Lines 182-184: Removed f-string prefix (plain strings)

### 3. Fixed super() Call
- Line 74: `super().__init__(daemon=True)` → `super(BackgroundMusic, self).__init__()` + `self.daemon = True`
  - Python 3.6 requires explicit class name in super()
  - daemon parameter set separately

## Python 3.6 Compatibility Checklist

✅ No f-strings (all converted to .format())
✅ super() calls use explicit class name
✅ No dataclasses
✅ No typing features from Python 3.7+ (using only typing.List, Tuple, Optional)
✅ No subprocess.DEVNULL issues (available in Python 3.3+)
✅ pathlib available (Python 3.4+)

## Testing

Run on Python 3.6:
```bash
# Jetson Nano (GStreamer camera)
python3.6 src/emoji_reactor/app.py

# PC/Mac (webcam)
python3.6 src/emoji_reactor/app.py --no-gstreamer --camera 0
```

## Dependencies
All dependencies support Python 3.6:
- opencv-python
- numpy
- ultralytics (YOLO11)
- mediapipe
- pathlib (built-in)
