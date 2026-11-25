Current setup takes an image and determines if a blink has occurred. If a blink is detected, it writes `{"time": blink_time}` to MongoDB. The ML service listens on port 8000.

The ML should be fed a continuous stream of images at a high rate so that it can detect blinks reliably.

Example usage would be like:

build image
```bash
cd machine-learning-client
docker build -t blink-ml .
```
run container
```bash
docker run --name blink-ml-container -p 8000:8000 blink-ml
```
web app
```python
with open("frame.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/process-frame", files={"image": f})
```