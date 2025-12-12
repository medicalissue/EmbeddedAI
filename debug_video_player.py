#!/usr/bin/env python3
"""
Debug script to find where illegal instruction occurs
"""

import sys

print("=" * 60)
print("Video Player Debug")
print("=" * 60)

# Test 1: Basic imports
print("\n[Test 1] Importing sys, os...")
import os
print("  OK")

# Test 2: Import cv2
print("\n[Test 2] Importing cv2...")
try:
    import cv2
    print("  OK - cv2 version: {}".format(cv2.__version__))
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 3: Import argparse
print("\n[Test 3] Importing argparse...")
try:
    import argparse
    print("  OK")
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 4: Create VideoCapture (without opening file)
print("\n[Test 4] Creating VideoCapture object...")
try:
    cap = cv2.VideoCapture()
    print("  OK")
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 5: Open a video file
print("\n[Test 5] Opening video file...")
if len(sys.argv) < 2:
    print("  SKIPPED - No video file provided")
    print("\nUsage: python3 debug_video_player.py <video_file>")
    sys.exit(0)

video_path = sys.argv[1]

if not os.path.isfile(video_path):
    print("  FAILED - File not found: {}".format(video_path))
    sys.exit(1)

try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  FAILED - Cannot open video file")
        sys.exit(1)
    print("  OK")
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 6: Get video properties
print("\n[Test 6] Getting video properties...")
try:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("  OK - {}x{}, {:.2f} fps, {} frames".format(width, height, fps, total_frames))
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 7: Read first frame
print("\n[Test 7] Reading first frame...")
try:
    ret, frame = cap.read()
    if not ret:
        print("  FAILED - Cannot read frame")
        sys.exit(1)
    print("  OK - Frame shape: {}".format(frame.shape))
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 8: Create window
print("\n[Test 8] Creating window...")
try:
    cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
    print("  OK")
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 9: Resize window
print("\n[Test 9] Resizing window...")
try:
    cv2.resizeWindow('Test', width, height)
    print("  OK")
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 10: Display frame
print("\n[Test 10] Displaying frame...")
try:
    cv2.imshow('Test', frame)
    print("  OK")
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Test 11: Wait for key
print("\n[Test 11] Waiting for key (press any key)...")
try:
    cv2.waitKey(2000)  # Wait 2 seconds
    print("  OK")
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

# Cleanup
print("\n[Cleanup] Releasing resources...")
try:
    cap.release()
    cv2.destroyAllWindows()
    print("  OK")
except Exception as e:
    print("  FAILED: {}".format(e))
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
