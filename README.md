---
title: AI B-Roll Generator
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# AI B-Roll Generator

Upload a talking-head MP4 → get a polished edit with b-roll cutaways, automatically.

**How it works:**
1. Groq Whisper transcribes your audio in seconds
2. Groq LLaMA plans which moments get b-roll vs face cam
3. Pexels provides free portrait 9:16 stock clips
4. ffmpeg composites the final video — your original audio runs throughout

**You need:** A free [Groq API key](https://console.groq.com)
