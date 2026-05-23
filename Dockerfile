FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake Whisper model so first request is instant
ENV HF_HOME=/app/.cache
RUN python3 -c "\
from faster_whisper import WhisperModel; \
m = WhisperModel('base', device='cpu', compute_type='int8'); \
print('Whisper model ready')"

COPY . .

RUN mkdir -p uploads jobs

EXPOSE 8080
ENV PORT=8080

CMD ["python", "app.py"]
