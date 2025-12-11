"""
Emoji Reactor - Hand & Face Tracking

States:
- HANDS_UP      : hand above --raise-thresh
- SMILING       : mouth aspect ratio > --smile-thresh
- STRAIGHT_FACE : default

Run:
  python app.py --no-gstreamer --camera 0   # PC/Mac
  python app.py                              # Jetson Nano (GStreamer)
"""

import argparse
import os
import sys
import time
import threading
import signal
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ASSETS = ROOT / "assets"
EMOJI_DIR = ASSETS / "emojis"
AUDIO_DIR = ASSETS / "audio"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_tracking import HandTrackingPipeline, draw_landmarks

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480


def load_emojis():
    """Load emoji images."""
    file_map = {
        "SMILING": "smile.jpg",
        "STRAIGHT_FACE": "plain.png",
        "HANDS_UP": "air.jpg",
    }

    loaded = {}
    for state, filename in file_map.items():
        path = EMOJI_DIR / filename
        img = cv2.imread(str(path))
        if img is not None:
            loaded[state] = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            print(f"[Warning] Could not load {path}")

    return loaded


def is_hand_up(landmarks, frame_h, thresh):
    """Check if wrist is above threshold."""
    return landmarks[0, 1] / frame_h < thresh




class BackgroundMusic(threading.Thread):
    """
    Jetson Nano optimized background music player using aplay.
    Supports pause/resume via SIGSTOP/SIGCONT signals.
    """
    def __init__(self, path, device="hw:0,3"):
        super().__init__(daemon=True)
        self.path = path
        self.device = device  # hw:0,3 for HDMI, hw:1,0 for expansion board
        self._running = True
        self._paused = False
        self._proc = None

    def pause(self):
        """Pause playback using SIGSTOP."""
        if self._proc and not self._paused:
            try:
                self._proc.send_signal(signal.SIGSTOP)
                self._paused = True
            except:
                pass

    def resume(self):
        """Resume playback using SIGCONT."""
        if self._proc and self._paused:
            try:
                self._proc.send_signal(signal.SIGCONT)
                self._paused = False
            except:
                pass

    def stop(self):
        """Stop playback and cleanup."""
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
            except:
                try:
                    self._proc.kill()
                except:
                    pass
        # Cleanup any remaining aplay processes
        os.system("killall -q -9 aplay 2>/dev/null")

    def run(self):
        """Main playback loop."""
        if not os.path.isfile(self.path):
            print(f"[BackgroundMusic] File not found: {self.path}")
            return

        # Kill any existing aplay processes
        os.system("killall -q -9 aplay 2>/dev/null")

        while self._running:
            try:
                # Start aplay process
                self._proc = subprocess.Popen(
                    ["aplay", "-D", self.device, "-q", self.path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Monitor process and handle pause/resume
                while self._running and self._proc.poll() is None:
                    # Handle pause state
                    if self._paused:
                        try:
                            self._proc.send_signal(signal.SIGSTOP)
                            # Wait until resume
                            while self._paused and self._running:
                                time.sleep(0.1)
                            # Resume
                            if self._running:
                                self._proc.send_signal(signal.SIGCONT)
                        except:
                            break
                    time.sleep(0.1)

                # Wait for process to finish (natural end of file)
                if self._proc:
                    self._proc.wait()

            except Exception as e:
                print(f"[BackgroundMusic] Error: {e}")
                break

            # Loop: restart playback if still running
            if not self._running:
                break


def play_sound(sound_name, device="hw:0,3"):
    """
    Play sound effect for emoji state change using aplay.

    Args:
        sound_name: Name of sound file (without extension)
        device: ALSA device (hw:0,3 for HDMI, hw:1,0 for expansion board)
    """
    sound_path = AUDIO_DIR / f"{sound_name}.wav"

    # Try .wav first, fallback to .mp3
    if not os.path.isfile(sound_path):
        sound_path = AUDIO_DIR / f"{sound_name}.mp3"

    if not os.path.isfile(sound_path):
        return

    try:
        # Use aplay for Jetson Nano
        subprocess.Popen(
            ["aplay", "-D", device, "-q", str(sound_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32')
    parser.add_argument('--prune', type=float, default=0.0, help='Pruning rate (0.0-0.7)')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--raise-thresh', type=float, default=0.25)
    parser.add_argument('--smile-thresh', type=float, default=0.35)
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--no-gstreamer', action='store_true')
    parser.add_argument('--audio-device', type=str, default='hw:0,3',
                        help='ALSA audio device (hw:0,3 for HDMI, hw:1,0 for expansion board)')
    args = parser.parse_args()

    emojis = load_emojis()
    blank_emoji = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    # Background music (aplay with SIGSTOP/SIGCONT support)
    bg_music_path = AUDIO_DIR / "yessir.wav"
    if not bg_music_path.exists():
        bg_music_path = AUDIO_DIR / "yessir.mp3"

    music = BackgroundMusic(str(bg_music_path), device=args.audio_device)
    music.start()
    print(f"[Audio] Background music: {bg_music_path.name} on {args.audio_device}")

    # Camera (GStreamer for Jetson Nano)
    if not args.no_gstreamer:
        pipeline = (
            "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=0"
        )
        print("Opening camera (GStreamer)...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        print(f"Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow('Reactor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Reactor', WINDOW_WIDTH * 2, WINDOW_HEIGHT)

    # YOLO11n-Pose for hands + MediaPipe Face Mesh
    print("[Init] YOLO11n-Pose (hands) + MediaPipe (face)...")
    pipeline = HandTrackingPipeline(precision=args.precision, prune_rate=args.prune)
    pipeline.print_stats()

    fps_hist = []
    prev_state = None

    print("\n[Ready] Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not args.no_mirror:
            frame = frame[:, ::-1].copy()
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        h, w = frame.shape[:2]

        # Hand & Face inference
        t0 = time.time()
        landmarks, detections, mar, mouth_center = pipeline.process_frame(frame)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        fps_hist = (fps_hist + [fps])[-30:]

        # State decision
        state = "STRAIGHT_FACE"
        if any(is_hand_up(lm, h, args.raise_thresh) for lm in landmarks):
            state = "HANDS_UP"
        elif mar > args.smile_thresh:
            state = "SMILING"

        # Play sound when state changes
        if state != prev_state and prev_state is not None:
            # Sound files should be named: HANDS_UP.wav/mp3, SMILING.wav/mp3, STRAIGHT_FACE.wav/mp3
            play_sound(state, device=args.audio_device)
        prev_state = state

        # Get emoji image
        emoji = emojis.get(state, blank_emoji)
        emoji_char = {"HANDS_UP": "🙌", "SMILING": "😊", "STRAIGHT_FACE": "😐"}.get(state, "❓")

        # Draw
        vis = frame.copy()
        for lm in landmarks:
            draw_landmarks(vis, lm)

        cv2.putText(vis, f"{state} {emoji_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"FPS {np.mean(fps_hist):.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Reactor', np.hstack((vis, emoji)))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    music.stop()


if __name__ == "__main__":
    main()
