import os
import json
import subprocess
import requests

PEXELS_API_KEY = "LppRQMMFSN0E7avfYQUoeQVdifahsxZwB0Uzag6Z7OQhZvemwAYfQ5eu"
GROQ_BASE = "https://api.groq.com/openai/v1"


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


def transcribe_audio(audio_path, api_key, duration):
    """Transcribe via Groq Whisper — runs in 1-3 seconds on their LPU chips."""
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{GROQ_BASE}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json"},
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()

    transcript = [
        {
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "text": s["text"].strip(),
        }
        for s in data.get("segments", [])
        if s.get("text", "").strip()
    ]

    if not transcript:
        transcript = [{"start": 0.0, "end": duration, "text": "(no speech detected)"}]

    return transcript


def analyze_segments(transcript, duration, api_key):
    lines = "\n".join(
        f"[{s['start']:.1f}s-{s['end']:.1f}s]: {s['text']}"
        for s in transcript
    )

    prompt = (
        "You are a professional video editor. Analyze this transcript and create an edit plan.\n\n"
        f"TRANSCRIPT:\n{lines}\n\n"
        "RULES:\n"
        '1. face_cam: hooks (opening ~10s), CTAs, emotional/personal moments, direct address\n'
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
        f'  {{"start": 18.5, "end": {duration:.1f}, "type": "face_cam"}}\n'
        "]\n\n"
        f"The last segment MUST end at exactly {duration:.1f}"
    )

    resp = requests.post(
        f"{GROQ_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    segments = json.loads(text.strip())

    if segments:
        segments[-1]["end"] = duration

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["end"] - seg["start"] < 3.0:
            last["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def search_pexels_video(keyword):
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": keyword, "orientation": "portrait", "per_page": 10, "size": "medium"}
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers=headers, params=params, timeout=30
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if not videos:
        return None

    for video in videos:
        for vf in sorted(video.get("video_files", []),
                         key=lambda x: x.get("height", 0), reverse=True):
            if vf.get("width", 9999) < vf.get("height", 0):
                return vf["link"]

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

    progress_cb(3, "Reading video info…")
    width, height, duration = get_video_info(video_path)

    progress_cb(8, "Extracting audio…")
    audio_path = f"{work}/audio.mp3"
    extract_audio(video_path, audio_path)

    progress_cb(15, "Transcribing with Groq Whisper…")
    transcript = transcribe_audio(audio_path, api_key, duration)

    progress_cb(40, "Planning b-roll segments with Groq LLaMA…")
    segments = analyze_segments(transcript, duration, api_key)

    with open(f"{work}/edit_plan.json", "w") as _f:
        json.dump(segments, _f)

    broll_clips = {}
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
                cut_face_cam(video_path, seg["start"], dur, out, width, height)

        seg_files.append(out)

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
