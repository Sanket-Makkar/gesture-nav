"""Live gesture recognition demo."""

import time
import argparse
import cv2

from runtime import GestureRuntime
from model import WINDOW_SIZE


CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# hand skeleton connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

GREEN = (180, 230, 85)
WHITE = (240, 240, 240)
GRAY = (175, 175, 175)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# draw hand skeleton over camera feed
def draw_hand(frame, landmarks):
    if landmarks is None:
        return

    h, w = frame.shape[:2]
    points = [(int(x * w), int(y * h)) for x, y, z in landmarks]

    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], GREEN, 2, cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, point, 4, GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 2, WHITE, -1, cv2.LINE_AA)


# draw runtime stats and gesture prediction
def draw_panel(frame, runtime, fps):
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (282, 280), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, "GESTURE NAVIGATION", (25, 36),
                FONT, 0.5, GREEN, 1, cv2.LINE_AA)

    stats = {
        "FPS": f"{fps:.1f}",
        "Device": str(runtime.device).upper(),
        "Buffer": f"{len(runtime.landmark_window)}/{WINDOW_SIZE}",
        "Spatial": f"{runtime.latency.get('Spatial', 0):.2f} ms",
        "Temporal": f"{runtime.latency.get('Temporal', 0):.2f} ms",
        "Total": f"{runtime.latency.get('Total', 0):.2f} ms"
    }

    for i, (label, value) in enumerate(stats.items()):
        y = 65 + i * 23

        cv2.putText(frame, label, (25, y),
                    FONT, 0.45, GRAY, 1, cv2.LINE_AA)

        cv2.putText(frame, value, (170, y),
                    FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    cv2.line(frame, (25, 194), (269, 194), GRAY, 1)

    # display prediction once buffer is full
    if runtime.prediction is not None:
        label = runtime.prediction["label"]
        confidence = runtime.prediction["confidence"]

        cv2.putText(frame, label, (25, 220),
                    FONT, 0.65, WHITE, 1, cv2.LINE_AA)

        cv2.putText(frame, f"{confidence:.1%}", (25, 242),
                    FONT, 0.45, GREEN, 1, cv2.LINE_AA)

        # confidence bar
        cv2.rectangle(frame, (25, 253), (269, 262), GRAY, -1)
        cv2.rectangle(frame, (25, 253),
                      (25 + int(244 * confidence), 262), GREEN, -1)

    else:
        cv2.putText(frame, "Filling buffer...", (25, 225),
                    FONT, 0.5, GRAY, 1, cv2.LINE_AA)

    # runtime status
    h, w = frame.shape[:2]
    status = "Hand detected" if runtime.landmarks is not None else "No hand"

    cv2.putText(frame, status, (12, h - 15),
                FONT, 0.5, GREEN, 1, cv2.LINE_AA)

    if not runtime.trained:
        cv2.putText(frame, "UNTRAINED MODEL", (12, h - 38),
                    FONT, 0.5, (0, 180, 255), 1, cv2.LINE_AA)


def demo(device=None, camera_id=0):
    runtime = GestureRuntime(device=device)

    # initialize camera
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    frame_interval = 1 / CAMERA_FPS
    fps = 0

    try:
        if not camera.isOpened():
            raise RuntimeError("Unable to open webcam")

        while True:
            start = time.perf_counter()

            # grab next frame
            success, frame = camera.read()
            if not success:
                break

            # run inference
            runtime.process_frame(frame)

            # visualize results
            draw_hand(frame, runtime.landmarks)
            draw_panel(frame, runtime, fps)

            cv2.imshow("Gesture Navigation", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # cap the loop at 30 fps
            elapsed = time.perf_counter() - start
            time.sleep(max(0, frame_interval - elapsed))

            # measure and smooth actual fps
            current_fps = 1 / (time.perf_counter() - start)
            fps = 0.9 * fps + 0.1 * current_fps

    finally:
        camera.release()
        runtime.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"])
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    demo(device=args.device, camera_id=args.camera)