"""Flask API"""
import time
from flask import Flask, request, jsonify

from blink_recorder import BlinkRecorder, get_collection
from eye_tracker import EyeTracker

app = Flask(__name__)

eye_tracker = EyeTracker("face_landmarker.task")
recorder = BlinkRecorder(eye_tracker, get_collection())


@app.post("/process-frame")
def process_frame():
    """
    main entry point to ml client
    calls process_frame
    returns state (blink, open, close)
    """

    if "image" not in request.files:
        return jsonify({"error": "missing image"}), 400

    image_bytes = request.files["image"].read()

    try:
        state = recorder.process_frame(image_bytes, time.time())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"state": state})


@app.get("/health")
def health():
    """check ml server status"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
