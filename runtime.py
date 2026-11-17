"""Real-time gesture recognition pipeline."""

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from model import load_model, INPUT_DIM, WINDOW_SIZE, CLASSES
from landmarker import HandLandmarker

# where we store the model post training
MODEL_PATH = Path(__file__).resolve().parent / "weights/gesture_cnn.pt"


class GestureRuntime:
    def __init__(self, device=None):
        # use metal if available
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # load trained weights if available, otherwise run in untrained mode
        self.trained = MODEL_PATH.exists()
        self.model = load_model(
            MODEL_PATH if self.trained else None, self.device
        )

        # initialize landmarker and ring buffer
        self.landmarker = HandLandmarker()
        self.landmark_window = deque(maxlen=WINDOW_SIZE)

        # runtime state
        self.landmarks = None
        self.prediction = None
        self.latency = {}

    @torch.inference_mode()
    def process_frame(self, frame):
        start = time.perf_counter()

        # extract landmarks from incoming frame
        self.landmarks = self.landmarker.detect(frame)

        # update ring buffer, zero fill if no hand detected
        coordinates = (
            self.landmarks.flatten()
            if self.landmarks is not None
            else np.zeros(INPUT_DIM, dtype=np.float32)
        )
        self.landmark_window.append(coordinates)

        spatial_end = time.perf_counter()
        self.latency = {"Spatial": (spatial_end - start) * 1000}

        # wait until ring buffer is full before running temporal inference
        if len(self.landmark_window) < WINDOW_SIZE:
            return None

        # construct input tensor (batch, time, features)
        sequence = torch.from_numpy(np.stack(self.landmark_window))
        sequence = sequence.unsqueeze(0).to(self.device)

        # run inference over the entire updated window
        probabilities = self.model(sequence).softmax(dim=-1)[0].cpu()
        index = probabilities.argmax().item()

        # package prediction for the application
        self.prediction = {
            "label": CLASSES[index],
            "confidence": probabilities[index].item(),
            "probabilities": probabilities.tolist(),
            "hand_detected": self.landmarks is not None,
            "trained": self.trained
        }

        # record inference latency in ms
        end = time.perf_counter()
        self.latency.update({
            "Temporal": (end - spatial_end) * 1000,
            "Total": (end - start) * 1000
        })

        return self.prediction

    def close(self):
        self.landmarker.close()