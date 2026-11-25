FROM python:3.11-slim

# System libraries that OpenCV / MediaPipe need, plus curl for the model download
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so Docker caches this slow step
RUN pip install --no-cache-dir pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

# Bake the MediaPipe model file into the image
RUN curl -L -o face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]