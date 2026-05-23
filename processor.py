import os
import json
import subprocess
import requests
import base64
import anthropic

PEXELS_API_KEY = "LppRQMMFSN0E7avfYQUoeQVdifahsxZwB0Uzag6Z7OQhZvemwAYfQ5eu"


def _make_client(api_key: str) -> anthropic.Anthropic:
    """Return an Anthropic client. Supports both regular API keys (sk-ant-api*)
    and OAuth/session tokens (sk-ant-si* / sk-ant-oa*) used in managed envs."""
    if api_key.startswith("sk-ant-api"):
        return anthropic.Anthropic(api_key=api_key)
    # OAuth bearer token (e.g. Claude Code managed environment)
    return anthropic.Anthropic(auth_token=api_key)


def run_ffmpeg(*args, error_label="ffmpeg"):
    result = subprocess.run(
        ["ffmpeg", "-y"] + list(args),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"{error_label} failed:\n{result.stderr[-2000:]}")
    return result


def get_video_info(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    width = height = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream["width"]
            height = stream["height"]
            break
    duration = float(data.get("format", {}).get("duration", 0))
    if not width or not duration:
        raise ValueError("Could not read video dimensions or duration")
    return width, height, duration


def extract_audio(video_path, audio_path):
    run_ffmpeg(
        "-i", video_path,
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", "-b:a", "32k",
        audio_path,
        error_label="audio extraction"
    )


def transcribe_audio(audio_path, _api_key, duration):
    """Transcribe using OpenAI Whisper (open-source, no API key needed).
    Whisper gives accurate word-level timestamps; Claude handles editorial analysis."""
    from faster_whisper import WhisperModel
    import os

    cache_dir = os.environ.get("HF_HOME", None)
    model = WhisperModel("base", device="cpu", compute_type="int8",
                         download_root=cache_dir)
    segments_iter, _ = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    # Group whisper word-level segments into sensible sentence chunks (3-20s)
    transcript = []
    chunk_start = None
    chunk_words = []
    chunk_end = 0.0

    for seg in segments_iter:
        words = seg.words if seg.words else []
        for w in words:
            if chunk_start is None:
                chunk_start = w.start
            chunk_words.append(w.word)
            chunk_end = w.end
            # Split on sentence-end punctuation or when chunk reaches ~15s
            is_sentence_end = w.word.strip().endswith((".", "!", "?", ","))
            chunk_duration = chunk_end - chunk_start
            if is_sentence_end and chunk_duration >= 3.0 or chunk_duration >= 15.0:
                transcript.append({
                    "start": round(chunk_start, 2),
                    "end": round(chunk_end, 2),
                    "text": "".join(chunk_words).strip(),
                })
                chunk_start = None
                chunk_words = []

    # Flush remaining words
    if chunk_words and chunk_start is not None:
        transcript.append({
            "start": round(chunk_start, 2),
            "end": round(duration, 2),
            "text": "".join(chunk_words).strip(),
        })
    elif transcript:
        transcript[-1]["end"] = round(duration, 2)

    if not transcript:
        # Fallback if audio was silent / too short
        transcript = [{"start": 0.0, "end": duration, "text": "(no speech detected)"}]

    return transcript


def analyze_segments(transcript, duration, api_key):
    client = _make_client(api_key)

    lines = "\n".join(
        f"[{s['start']:.1f}s-{s['end']:.1f}s]: {s['text']}"
        for s in transcript
    )

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": (
                "You are a professional video editor. Analyze this transcript and create an edit plan.\n\n"
                f"TRANSCRIPT:\n{lines}\n\n"
                "RULES:\n"
                '1. face_cam: hooks (opening ~10s), CTAs, emotional/personal moments, direct address ("you", "I")\n'
                "2. broll: describing products, places, concepts, tips, steps — anything visual\n"
                "3. Minimum 3 seconds per segment\n"
                "4. Aim for 40-60% b-roll coverage\n"
                f"5. Segments must be contiguous, covering 0.0 to {duration:.1f}s exactly\n"
                "6. For broll: 2-4 word Pexels keyword (concrete, visual, searchable)\n"
                "7. First segment is almost always face_cam (the hook)\n"
                "8. Last segment is often face_cam (the CTA)\n\n"
                "Return ONLY a JSON array, no markdown:\n"
                '[\n'
                '  {"start": 0.0, "end": 9.0, "type": "face_cam"},\n'
                '  {"start": 9.0, "end": 18.5, "type": "broll", "keyword": "morning coffee laptop"},\n'
                '  {"start": 18.5, "end": 25.0, "type": "face_cam"},\n'
                f'  {{"start": 25.0, "end": {duration:.1f}, "type": "face_cam"}}\n'
                "]\n\n"
                f"The last segment MUST end at exactly {duration:.1f}"
            )
        }]
    )

    text = resp.content[0].text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    segments = json.loads(text.strip())

    # Enforce exact duration at the end
    if segments:
        segments[-1]["end"] = duration

    # Merge adjacent same-type segments that would be under 3 seconds
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        seg_dur = seg["end"] - seg["start"]
        if seg_dur < 3.0:
            last["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def search_pexels_video(keyword):
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": keyword,
        "orientation": "portrait",
        "per_page": 10,
        "size": "medium",
    }
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers=headers, params=params, timeout=30
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if not videos:
        return None

    # Prefer genuine portrait files (width < height)
    for video in videos:
        for vf in sorted(
            video.get("video_files", []),
            key=lambda x: x.get("height", 0), reverse=True
        ):
            if vf.get("width", 9999) < vf.get("height", 0):
                return vf["link"]

    # Fallback: first available file
    for video in videos:
        files = video.get("video_files", [])
        if files:
            return files[0]["link"]
    return None


def download_video(url, dest):
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)


def cut_face_cam(original, start, dur, out, w, h):
    run_ffmpeg(
        "-ss", str(start), "-i", original,
        "-t", str(dur),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        out, error_label="face-cam cut"
    )


def cut_broll(broll_path, dur, out, w, h):
    run_ffmpeg(
        "-stream_loop", "-1", "-i", broll_path,
        "-t", str(dur),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,"
               f"crop={w}:{h}",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        out, error_label="b-roll cut"
    )


def process_video(video_path, job_id, api_key, progress_cb):
    work = f"jobs/{job_id}"
    os.makedirs(work, exist_ok=True)

    # 1 — video info
    progress_cb(3, "Reading video info…")
    width, height, duration = get_video_info(video_path)

    # 2 — audio
    progress_cb(8, "Extracting audio…")
    audio_path = f"{work}/audio.mp3"
    extract_audio(video_path, audio_path)

    # 3 — transcribe (Whisper model downloads ~150 MB on first run)
    progress_cb(15, "Transcribing audio (first run downloads Whisper model ~150 MB)…")
    transcript = transcribe_audio(audio_path, api_key, duration)

    # 4 — plan edit
    progress_cb(40, "Planning b-roll segments with Claude…")
    segments = analyze_segments(transcript, duration, api_key)

    # Persist edit plan so the status endpoint can surface it
    with open(f"{work}/edit_plan.json", "w") as _f:
        json.dump(segments, _f)

    # 5 — download b-roll clips
    broll_clips = {}          # broll_index -> local path or None
    broll_segs = [s for s in segments if s["type"] == "broll"]
    n_broll = max(len(broll_segs), 1)

    for idx, seg in enumerate(broll_segs):
        keyword = seg.get("keyword", "nature landscape")
        pct = 50 + int(idx / n_broll * 20)
        progress_cb(pct, f'Downloading b-roll: "{keyword}"...')
        url = search_pexels_video(keyword)
        if url:
            clip = f"{work}/broll_{idx}.mp4"
            download_video(url, clip)
            broll_clips[idx] = clip
        else:
            broll_clips[idx] = None

    # 6 — cut every segment
    progress_cb(72, "Cutting segments…")
    seg_files = []
    broll_idx = 0

    for i, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        out = f"{work}/seg_{i:03d}.mp4"

        if seg["type"] == "face_cam":
            cut_face_cam(video_path, seg["start"], dur, out, width, height)
        else:
            clip = broll_clips.get(broll_idx)
            broll_idx += 1
            if clip:
                cut_broll(clip, dur, out, width, height)
            else:
                # fall back to face cam if Pexels had nothing
                cut_face_cam(video_path, seg["start"], dur, out, width, height)

        seg_files.append(out)

    # 7 — concatenate video-only stream
    progress_cb(85, "Joining clips…")
    concat_txt = f"{work}/concat.txt"
    with open(concat_txt, "w") as f:
        for sp in seg_files:
            f.write(f"file '{os.path.abspath(sp)}'\n")

    video_only = f"{work}/video_only.mp4"
    run_ffmpeg(
        "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-c:v", "copy",
        video_only, error_label="concat"
    )

    # 8 — mux with original audio
    progress_cb(93, "Adding original audio…")
    output = f"{work}/output.mp4"
    run_ffmpeg(
        "-i", video_only,
        "-i", video_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output, error_label="mux audio"
    )

    progress_cb(100, "Done!")
    return output
