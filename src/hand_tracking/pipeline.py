"""
Hand & Face Tracking Pipeline (Python 3.6+ Compatible)
- Hand: YOLO11n-Pose ONNX (21 keypoints) from chrismuntean/YOLO11n-pose-hands
- Face: MediaPipe Face Mesh (468 keypoints, focus on mouth for MAR)
"""

import cv2
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("[Error] onnxruntime not installed. Install with: pip3 install onnxruntime")
    raise

# Import MediaPipe solutions directly to avoid TensorFlow dependency issues
from mediapipe.python.solutions import face_mesh as mp_face_mesh

ROOT = Path(__file__).resolve().parents[2]

# Hand connections (21 keypoints) - MediaPipe format
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]

# MediaPipe Face Mesh mouth indices (outer lips)
MOUTH_UPPER_OUTER = 13
MOUTH_LOWER_OUTER = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


def draw_landmarks(frame, landmarks, color=(0, 255, 0)):
    """Draw hand landmarks on frame."""
    for pt in landmarks:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 5, color, -1)
    for i, j in HAND_CONNECTIONS:
        if i < len(landmarks) and j < len(landmarks):
            pt1 = (int(landmarks[i, 0]), int(landmarks[i, 1]))
            pt2 = (int(landmarks[j, 0]), int(landmarks[j, 1]))
            cv2.line(frame, pt1, pt2, color, 2)


def draw_detections(frame, detections, color=(255, 0, 0)):
    """Draw detection boxes on frame."""
    for det in detections:
        x1, y1, x2, y2 = det[:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image to new_shape while maintaining aspect ratio.
    Returns resized image, ratio, and padding.
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Non-Maximum Suppression for pose detection.

    Args:
        prediction: (batch_size, num_boxes, 68)
                   68 = 4 (box) + 1 (obj_conf) + 63 (21 keypoints * 3)
        conf_thres: confidence threshold
        iou_thres: IoU threshold
        max_det: maximum detections

    Returns:
        List of detections per image: (n, 68) where first 5 = (x1, y1, x2, y2, conf)
    """
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        # Filter by confidence
        x = x[x[:, 4] > conf_thres]

        if not x.shape[0]:
            output.append(np.zeros((0, 68)))
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections: (n, 68) = (xyxy, conf, keypoints)
        conf = x[:, 4:5]
        kpt = x[:, 5:]  # keypoints (63 values = 21 * 3)

        # Combine
        det = np.concatenate((box, conf, kpt), axis=1)

        # NMS
        boxes = det[:, :4]
        scores = det[:, 4]

        # Simple NMS implementation
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            conf_thres,
            iou_thres
        )

        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            det = det[indices.flatten()]

            if len(det) > max_det:
                det = det[:max_det]
        else:
            det = np.zeros((0, 68))

        output.append(det)

    return output


class HandTrackingPipeline:
    """Hand & Face tracking using YOLO ONNX + MediaPipe."""

    def __init__(self, precision="fp32", prune_rate=0.0):
        self.precision = precision
        self.prune_rate = prune_rate
        self.input_size = 640

        model_desc = "{}".format(precision)
        if prune_rate > 0:
            model_desc += ", pruned {}%".format(int(prune_rate*100))

        print("[Pipeline] Loading YOLO11n-Pose hand model ({})...".format(model_desc))

        # Select model based on precision and pruning
        if prune_rate > 0:
            # Use pruned model
            prune_pct = int(prune_rate * 100)
            model_path = ROOT / "assets/models/yolo11n_hand_pose_pruned_{}.onnx".format(prune_pct)
            if not model_path.exists():
                print("[Warning] Pruned ONNX model not found, using base model")
                model_path = ROOT / "assets/models/yolo11n_hand_pose.onnx"
        else:
            model_path = ROOT / "assets/models/yolo11n_hand_pose.onnx"

        if not model_path.exists():
            raise FileNotFoundError("ONNX model not found: {}\nRun: python3 export_yolo_to_onnx.py".format(model_path))

        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']

        # Try to use CUDA if available (Jetson Nano has CUDA)
        try:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("[ONNX] CUDA available, using GPU acceleration")
        except:
            pass

        self.session = ort.InferenceSession(str(model_path), providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        print("[Pipeline] YOLO11n-Pose ONNX loaded (21 keypoints per hand, {})".format(model_desc))
        print("[Pipeline] Input: {}, Output: {}".format(self.input_name, self.output_names[0]))

        print("[Pipeline] Initializing MediaPipe Face Mesh...")
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[Pipeline] MediaPipe Face Mesh loaded (468 keypoints)")

    def preprocess(self, img):
        """Preprocess image for YOLO ONNX model."""
        # Letterbox resize
        img_resized, ratio, (dw, dh) = letterbox(img, new_shape=(self.input_size, self.input_size))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img_norm = img_rgb.astype(np.float32) / 255.0

        # HWC to CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)

        return img_batch, ratio, (dw, dh)

    def process_frame(self, frame):
        """Process frame and return hand landmarks + face info."""
        h, w = frame.shape[:2]

        landmarks_list = []
        detections = []
        mar = 0.0
        mouth_center = None

        # === Hand Processing (YOLO ONNX) ===
        try:
            # Preprocess
            input_tensor, ratio, (dw, dh) = self.preprocess(frame)

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # Post-process
            # Output shape: [1, 56, 8400] for YOLO11n-pose
            # Transpose to [1, 8400, 56]
            prediction = outputs[0]
            if len(prediction.shape) == 3 and prediction.shape[1] < prediction.shape[2]:
                prediction = np.transpose(prediction, (0, 2, 1))

            # NMS
            detections_nms = non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45)

            # Process detections
            for det in detections_nms:
                if len(det) == 0:
                    continue

                for i in range(len(det)):
                    # Box
                    x1, y1, x2, y2 = det[i, :4]
                    conf = det[i, 4]

                    # Keypoints (51 values = 17 kpts * 3, but we need 21 kpts)
                    # YOLO11n-pose-hands should have 21 keypoints
                    # Shape: [63] = 21 * 3 (x, y, conf)
                    kpts_raw = det[i, 5:]  # Get all remaining values

                    # Reshape to [21, 3]
                    num_kpts = len(kpts_raw) // 3
                    if num_kpts >= 21:
                        kpts = kpts_raw[:63].reshape(21, 3)  # Take first 21 keypoints
                    else:
                        # Pad if needed
                        kpts = np.zeros((21, 3))
                        kpts[:num_kpts] = kpts_raw.reshape(num_kpts, 3)

                    # Scale keypoints back to original image
                    kpts_scaled = kpts.copy()
                    kpts_scaled[:, 0] = (kpts[:, 0] - dw) / ratio  # x
                    kpts_scaled[:, 1] = (kpts[:, 1] - dh) / ratio  # y

                    landmarks_list.append(kpts_scaled)

                    # Scale box back to original image
                    x1 = (x1 - dw) / ratio
                    y1 = (y1 - dh) / ratio
                    x2 = (x2 - dw) / ratio
                    y2 = (y2 - dh) / ratio

                    detections.append(np.array([x1, y1, x2, y2, conf]))

        except Exception as e:
            print("[Hand] Error: {}".format(e))

        # === Face Processing (MediaPipe) ===
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                # Extract mouth landmarks for MAR calculation
                upper = face_landmarks.landmark[MOUTH_UPPER_OUTER]
                lower = face_landmarks.landmark[MOUTH_LOWER_OUTER]
                left = face_landmarks.landmark[MOUTH_LEFT]
                right = face_landmarks.landmark[MOUTH_RIGHT]

                # Convert to pixel coordinates
                upper_px = np.array([upper.x * w, upper.y * h])
                lower_px = np.array([lower.x * w, lower.y * h])
                left_px = np.array([left.x * w, left.y * h])
                right_px = np.array([right.x * w, right.y * h])

                # Calculate MAR
                mouth_height = np.linalg.norm(upper_px - lower_px)
                mouth_width = np.linalg.norm(left_px - right_px)

                if mouth_width > 1:
                    mar = mouth_height / mouth_width

                mouth_center = (upper_px + lower_px + left_px + right_px) / 4

        except Exception as e:
            print("[Face] Error: {}".format(e))

        return landmarks_list, np.array(detections) if detections else np.array([]), mar, mouth_center

    def print_stats(self):
        print("\n" + "=" * 50)
        print("Hand & Face Tracking")
        print("=" * 50)
        print("Hand: YOLO11n-Pose ONNX (21 keypoints)")
        print("Face: MediaPipe Face Mesh (468 keypoints)")
        print("=" * 50 + "\n")

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
