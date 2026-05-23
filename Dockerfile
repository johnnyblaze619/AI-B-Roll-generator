FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake Whisper base model into the image so first run is instant
ENV HF_HOME=/app/.cache
RUN python3 -c "\
from faster_whisper import WhisperModel; \
m = WhisperModel('base', device='cpu', compute_type='int8'); \
print('Whisper model ready')"

# Copy app
COPY . .

RUN mkdir -p uploads jobs

# Hugging Face Spaces serves on 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
