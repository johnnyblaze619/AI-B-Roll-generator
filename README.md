# B-Roll Editor

Upload a talking-head MP4 and get a polished edit with automatic b-roll cutaways.

## How it works
1. Whisper transcribes your audio with accurate timestamps
2. Claude (Anthropic) plans which moments get face cam vs b-roll
3. Pexels provides free portrait 9:16 stock clips
4. ffmpeg composites the final video with your original audio throughout

## Setup
```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000 and enter your Anthropic API key.
