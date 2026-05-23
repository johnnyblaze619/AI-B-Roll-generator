#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Install Python dependencies if needed
pip3 install -r requirements.txt -q

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  echo "Installing ffmpeg..."
  apt-get install -y ffmpeg >/dev/null 2>&1 || \
    brew install ffmpeg 2>/dev/null || \
    { echo "ERROR: Please install ffmpeg (https://ffmpeg.org/download.html)"; exit 1; }
fi

echo ""
echo "==================================="
echo "  B-Roll Editor starting on :5000"
echo "==================================="
echo "  Open http://localhost:5000"
echo "==================================="
echo ""

python3 app.py
