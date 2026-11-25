"""ml client. send in frames, any blink is saved to MongoDB."""

import os
from datetime import datetime, timezone

from pymongo import MongoClient
from blink_detector import BlinkDetector


def get_collection():
    """connect to MongoDB and return the blink_record collection."""
    client = MongoClient(os.environ.get("MONGO_URI", "mongodb://mongodb:27017"))
    return client[os.environ.get("MONGO_DB", "blink_tracker")].blink_record


class BlinkRecorder:
    """analyzes frames one at a time and saves the time of each blink."""

    def __init__(self, eye_tracker, collection):
        self.eye_tracker = eye_tracker
        self.collection = collection
        self.blink_detector = BlinkDetector()

    def process_frame(self, image_bytes, timestamp):
        """
        analyze one jpeg frame (timestamp = epoch seconds)
        returns "blink", "open" or "close". saves {"time": ...} to the database whenever a blink finishes
        """
        ear = self.eye_tracker.measure(image_bytes)
        state, blink_time = self.blink_detector.update(ear, timestamp)
        if blink_time is not None:
            self.collection.insert_one(
                {"time": datetime.fromtimestamp(blink_time, tz=timezone.utc)}
            )
        return state