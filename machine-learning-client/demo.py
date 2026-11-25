"""
local demo
reads your webcam directly and shows a window instead of waiting for HTTP requests.
"""

import os
import time
import cv2

from blink_recorder import BlinkRecorder
from eye_tracker import EyeTracker, ensure_model

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "1"))




class ListCollection:
    """Stands in for a MongoDB collection so the demo needs no database."""

    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        """Same method name BlinkRecorder calls on a real collection."""
        self.docs.append(doc)


def put_text(frame, text, y, color=(255, 255, 255)):
    """Draw a line of text with a dark outline"""
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    """Run the webcam loop."""

    collection = ListCollection()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(
            f"Could not open camera {CAMERA_INDEX}. "
            "Try CAMERA_INDEX=0 or 2."
        )
        return

    path = ensure_model()

    with EyeTracker(path) as eye_tracker:
        recorder = BlinkRecorder(eye_tracker, collection)
        detector = recorder.blink_detector

        total_blinks = 0
        flash_until = 0.0

        while True:
            grabbed, frame = cap.read()

            if not grabbed:
                print("Could not read a frame from the camera")
                break

            frame = cv2.flip(frame, 1)

            # Send jpeg bytes
            encoded, jpeg = cv2.imencode(".jpg", frame)

            if not encoded:
                continue

            now = time.time()

            state = recorder.process_frame(
                jpeg.tobytes(),
                now
            )

            if state == "blink":
                total_blinks += 1
                flash_until = now + 0.3
                print(f"blinked total={total_blinks}")

            put_text(frame, f"state: {state}", 30)
            put_text(
                frame,
                f"threshold: {detector.threshold:.3f}",
                60
            )
            put_text(frame, f"blinks: {total_blinks}", 90)

            if now < flash_until:
                put_text(
                    frame,
                    "BLINK",
                    frame.shape[0] - 20,
                    (0, 0, 255)
                )

            cv2.imshow("Blink demo (q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
