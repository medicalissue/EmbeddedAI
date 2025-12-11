#!/usr/bin/env python3
"""
GStreamer HDMI 직접 출력 테스트 (Jetson Nano 전용)
cv2.imshow() 대신 nvoverlaysink 사용
"""

import cv2
import numpy as np
import time


def create_gstreamer_display_pipeline(width=640, height=480):
    """
    GStreamer pipeline for hardware-accelerated HDMI output.

    nvoverlaysink:
    - GPU에서 직접 HDMI로 출력
    - X11/GTK 우회
    - Zero-copy operation
    """
    pipeline = (
        f"appsrc ! "
        f"video/x-raw, format=BGR, width={width}, height={height}, framerate=30/1 ! "
        f"videoconvert ! "
        f"video/x-raw(memory:NVMM), format=NV12 ! "
        f"nvoverlaysink overlay-x=0 overlay-y=0 overlay-w={width} overlay-h={height}"
    )
    return pipeline


def test_gstreamer_hdmi():
    """Test GStreamer HDMI output vs cv2.imshow."""

    print("=" * 80)
    print("GStreamer HDMI Direct Output Test (Jetson Nano)")
    print("=" * 80)

    width, height = 640, 480

    # Method 1: cv2.imshow (baseline)
    print("\n[Method 1] cv2.imshow (X11/GTK)")
    times_cv2 = []

    for i in range(100):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        start = time.time()
        cv2.imshow('CV2 Test', frame)
        cv2.waitKey(1)
        times_cv2.append(time.time() - start)

    cv2.destroyAllWindows()

    print(f"  Average time: {np.mean(times_cv2)*1000:.2f} ms")
    print(f"  Max FPS: {1.0/np.mean(times_cv2):.1f}")

    # Method 2: GStreamer nvoverlaysink
    print("\n[Method 2] GStreamer nvoverlaysink (GPU direct)")

    try:
        gst_pipeline = create_gstreamer_display_pipeline(width, height)
        out = cv2.VideoWriter(
            gst_pipeline,
            cv2.CAP_GSTREAMER,
            0,  # fourcc (ignored for appsrc)
            30,  # fps
            (width, height),
            True
        )

        if not out.isOpened():
            print("  ✗ Failed to open GStreamer pipeline")
            print("  This requires Jetson Nano with nvoverlaysink support")
            return

        times_gst = []

        for i in range(100):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            start = time.time()
            out.write(frame)
            times_gst.append(time.time() - start)

        out.release()

        print(f"  Average time: {np.mean(times_gst)*1000:.2f} ms")
        print(f"  Max FPS: {1.0/np.mean(times_gst):.1f}")

        # Comparison
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"cv2.imshow:        {np.mean(times_cv2)*1000:.2f} ms")
        print(f"GStreamer HDMI:    {np.mean(times_gst)*1000:.2f} ms")
        print(f"Speedup:           {np.mean(times_cv2)/np.mean(times_gst):.1f}x faster")
        print(f"Overhead reduced:  {(np.mean(times_cv2)-np.mean(times_gst))*1000:.2f} ms")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  GStreamer nvoverlaysink requires Jetson Nano")


def test_camera_with_gstreamer_display():
    """
    Full example: Camera input + GStreamer HDMI output.
    최소 오버헤드로 카메라 → HDMI 직접 출력
    """
    print("\n" + "=" * 80)
    print("Camera → GStreamer HDMI Pipeline")
    print("=" * 80)

    width, height = 640, 480

    # Camera input (GStreamer)
    camera_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

    # Display output (GStreamer)
    display_pipeline = create_gstreamer_display_pipeline(width, height)

    try:
        cap = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)
        out = cv2.VideoWriter(display_pipeline, cv2.CAP_GSTREAMER, 0, 30, (width, height), True)

        if not cap.isOpened() or not out.isOpened():
            print("  ✗ Failed to open camera or display pipeline")
            return

        print("\n  Press Ctrl+C to stop")
        print("  Monitoring FPS...\n")

        frame_times = []
        frame_count = 0

        while True:
            start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Process frame here (hand tracking, etc.)
            # ...

            # Write to HDMI (GPU-accelerated)
            out.write(frame)

            frame_times.append(time.time() - start)
            frame_count += 1

            if frame_count % 30 == 0:
                avg_fps = 1.0 / np.mean(frame_times[-30:])
                print(f"  FPS: {avg_fps:.1f}")

            if frame_count >= 300:  # Test for 10 seconds at 30 FPS
                break

        cap.release()
        out.release()

        print(f"\n  Average FPS: {1.0/np.mean(frame_times):.1f}")
        print(f"  Average frame time: {np.mean(frame_times)*1000:.2f} ms")

    except KeyboardInterrupt:
        print("\n  Stopped by user")
    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    # Test 1: Display overhead comparison
    test_gstreamer_hdmi()

    # Test 2: Full camera pipeline (uncomment on Jetson Nano)
    # test_camera_with_gstreamer_display()
