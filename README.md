---
title: B-Roll Editor
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# B-Roll Editor

Upload a talking-head MP4 → get a polished edit with b-roll cutaways, automatically.

**How it works:**
1. Whisper transcribes your audio with accurate timestamps
2. Claude (Anthropic) reads the transcript and plans which moments get b-roll vs face cam
3. Pexels provides free portrait 9:16 stock clips for each b-roll segment
4. ffmpeg composites the final video — your original audio runs throughout

**You need:** An [Anthropic API key](https://console.anthropic.com)
