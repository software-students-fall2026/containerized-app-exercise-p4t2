"""
mediapipe wrapper
image bytes in, eye aspect ratio out (EAR)
"""

import threading
import math
import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# mediapipe face mesh landmark indices
# [outer corner, top 1, top 2, inner corner, bottom 2, bottom 1]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def _eye_points(landmarks, indices, width, height):
    """Convert normalized landmarks to pixel (x, y) tuples."""
    return [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]


def eye_aspect_ratio(points):
    """
    calc EAR from 6 points
    eye height divided by eye width

    outer corner, top 1, top 2, inner corner,
    bottom 2, bottom 1.
    """
    vertical = math.dist(points[1], points[5]) + math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])
    return vertical / (2.0 * horizontal)

class EyeTracker:
    """finds a face in an image and measures how open the eyes are"""

    def __init__(self, model_path):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()  # one frame at a time through MediaPipe

    def measure(self, image_bytes):
        """
        Return the average EAR of both eyes, or None if no face is found.
        Raises ValueError if image_bytes is not a decodable image.
        """
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("not a valid image")
        height, width = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        right = _eye_points(landmarks, RIGHT_EYE, width, height)
        left = _eye_points(landmarks, LEFT_EYE, width, height)
        return (eye_aspect_ratio(right) + eye_aspect_ratio(left)) / 2.0

    def close(self):
        """Free resources"""
        with self._lock:
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()





def ensure_model(path="face_landmarker.task",
                 url="https://storage.googleapis.com/mediapipe-models/"
                 "face_landmarker/face_landmarker/float16/1/face_landmarker.task"):
    """
    download ml model if it doesn't exist
    currently only used in demo.py, for containers the download is in the dockerfile
    """
    if not os.path.exists(path):
        print("Downloading face landmarker model...")
        urllib.request.urlretrieve(url, path)
    return path
