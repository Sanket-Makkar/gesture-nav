# Gesture Navigation

Lightweight dynamic hand gesture recognition for real-time touchless app navigation.

## Architecture

Webcam -> MediaPipe -> Sliding Window -> Temporal CNN -> Prediction

## Project Structure

- model.py: Neural network architecture and factories
- landmarker.py: MediaPipe landmark extraction
- runtime.py: Inference pipeline
- demo.py: Standalone webcam demonstration
- train.ipynb: Offline training and evaluation

## Setup

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run

    python demo.py

## Device Selection

    python demo.py --device cpu
    python demo.py --device mps

Press q to quit.

MediaPipe weights are downloaded automatically.
The gesture classifier loads weights/gesture_cnn.pt if available.
Without trained weights, gesture predictions are meaningless.

## Training

Use train.ipynb locally or in Google Colab.
