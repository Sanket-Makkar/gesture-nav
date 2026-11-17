"""MediaPipe hand landmark extraction."""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


WEIGHTS_PATH = Path(__file__).resolve().parent / "weights/hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


class HandLandmarker:

    def __init__(self):
        WEIGHTS_PATH.parent.mkdir(exist_ok=True)

        if not WEIGHTS_PATH.exists():
            print("Downloading MediaPipe weights...")
            temporary = WEIGHTS_PATH.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, temporary)
            temporary.replace(WEIGHTS_PATH)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(WEIGHTS_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1
        )

        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )
        self.last_timestamp = -1

    def detect(self, frame):
        """Return a (21, 3) landmark array or None."""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = max(
            time.monotonic_ns() // 1_000_000,
            self.last_timestamp + 1
        )
        self.last_timestamp = timestamp

        result = self.detector.detect_for_video(image, timestamp)

        if not result.hand_landmarks:
            return None

        return np.array(
            [[p.x, p.y, p.z] for p in result.hand_landmarks[0]],
            dtype=np.float32
        )

    def close(self):
        self.detector.close()
