#!/usr/bin/env python3
"""
Simple Video Player with Loop
- Plays video file in infinite loop
- Press 'q' or ESC to quit
- Press SPACE to pause/resume
"""

import cv2
import argparse
import sys
import os


def play_video(video_path, window_name="Video Player"):
    """
    Play video file in infinite loop.

    Args:
        video_path: Path to video file
        window_name: OpenCV window name
    """
    if not os.path.isfile(video_path):
        print("[Error] Video file not found: {}".format(video_path))
        return False

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[Error] Cannot open video file: {}".format(video_path))
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 60)
    print("Video Player")
    print("=" * 60)
    print("File: {}".format(video_path))
    print("Resolution: {}x{}".format(width, height))
    print("FPS: {:.2f}".format(fps))
    print("Total Frames: {}".format(total_frames))
    print("Duration: {:.2f} seconds".format(total_frames / fps if fps > 0 else 0))
    print("=" * 60)
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  'q' or ESC - Quit")
    print("=" * 60)

    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # Calculate delay between frames (in milliseconds)
    delay = int(1000 / fps) if fps > 0 else 30

    paused = False
    frame_count = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()

                # If end of video, restart from beginning
                if not ret:
                    print("\n[Loop] Restarting video...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    continue

                frame_count += 1

                # Display frame number and timestamp
                timestamp = frame_count / fps if fps > 0 else 0
                info_text = "Frame: {}/{} | Time: {:.2f}s".format(
                    frame_count, total_frames, timestamp
                )
                cv2.putText(
                    frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                # Show frame
                cv2.imshow(window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n[Exit] User quit")
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                if paused:
                    print("[Paused]")
                else:
                    print("[Resumed]")

    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("[Exit] Video player closed")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Simple video player with infinite loop"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to video file (mp4, avi, mov, etc.)"
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Video Player",
        help="Window title (default: Video Player)"
    )

    args = parser.parse_args()

    # Play video
    success = play_video(args.video, args.window_name)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
