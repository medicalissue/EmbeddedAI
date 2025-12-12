#!/usr/bin/env python3
"""
Export YOLO11n-Pose model to ONNX format for Python 3.6 compatibility.

Run this on a system with Python 3.8+ and ultralytics installed.
Then copy the .onnx file to Jetson Nano.

Usage:
    python3 export_yolo_to_onnx.py
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "assets" / "models"

def export_model(model_path, output_name):
    """Export YOLO model to ONNX format."""
    print("=" * 80)
    print("Exporting: {}".format(model_path.name))
    print("=" * 80)

    if not model_path.exists():
        print("[Error] Model not found: {}".format(model_path))
        return False

    try:
        # Load model
        model = YOLO(str(model_path))

        # Export to ONNX with optimizations
        success = model.export(
            format="onnx",
            imgsz=640,
            simplify=True,  # ONNX simplification
            opset=12,  # ONNX opset version (compatible with older ONNX Runtime)
            dynamic=False,  # Fixed input size for better performance
        )

        # Rename to consistent name
        exported_path = model_path.parent / (model_path.stem + ".onnx")
        final_path = MODEL_DIR / output_name

        if exported_path.exists():
            if final_path.exists():
                final_path.unlink()
            exported_path.rename(final_path)
            print("[Success] Exported to: {}".format(final_path))
            print("          File size: {:.2f} MB".format(final_path.stat().st_size / 1024 / 1024))
            return True
        else:
            print("[Error] Export failed")
            return False

    except Exception as e:
        print("[Error] {}".format(e))
        return False


def main():
    print("\n" + "=" * 80)
    print("YOLO11n-Pose to ONNX Converter")
    print("Python 3.6+ Compatible Format")
    print("=" * 80 + "\n")

    # Export base model
    base_model = MODEL_DIR / "yolo11n_hand_pose.pt"
    if base_model.exists():
        export_model(base_model, "yolo11n_hand_pose.onnx")
    else:
        print("[Warning] Base model not found: {}".format(base_model))

    # Export pruned models if they exist
    for prune_rate in [10, 20, 30, 40, 50]:
        pruned_model = MODEL_DIR / "yolo11n_hand_pose_pruned_{}.pt".format(prune_rate)
        if pruned_model.exists():
            print("\n")
            export_model(pruned_model, "yolo11n_hand_pose_pruned_{}.onnx".format(prune_rate))

    print("\n" + "=" * 80)
    print("Export Complete")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Copy .onnx files to Jetson Nano")
    print("2. Install ONNX Runtime on Jetson Nano:")
    print("   pip3 install onnxruntime")
    print("3. Run the app with Python 3.6+")
    print()


if __name__ == "__main__":
    main()
