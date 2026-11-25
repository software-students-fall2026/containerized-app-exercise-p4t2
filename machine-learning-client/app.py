from flask import Flask, request, jsonify
import time

from blink_recorder import BlinkRecorder, get_collection
from eye_tracker import EyeTracker

app = Flask(__name__)

eye_tracker = EyeTracker("face_landmarker.task")
recorder = BlinkRecorder(
    eye_tracker,
    get_collection()
)



'''
main entry point
calls process_frame
returns state (blink, open, close)
'''
@app.post("/process-frame")
def process_frame():
    if "image" not in request.files:
        return jsonify({"error": "missing image"}), 400

    image_bytes = request.files["image"].read()

    try:
        state = recorder.process_frame(
            image_bytes,
            time.time()
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "state": state
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)