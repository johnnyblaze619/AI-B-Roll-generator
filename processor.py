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


def _groq_llm(prompt, api_key):
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
    return text.strip()


def analyze_segments(transcript, duration, api_key):
    lines = "\n".join(
        f"[{s['start']:.1f}s-{s['end']:.1f}s]: {s['text']}"
        for s in transcript
    )

    prompt = (
        "You are a professional video editor. Create a b-roll edit plan with a BEAUTIFUL RHYTHM.\n\n"
        f"VIDEO TRANSCRIPT ({duration:.0f} seconds total):\n{lines}\n\n"
        "RHYTHM RULES — follow exactly:\n"
        "1. Segment 1: face_cam, 8-12s  ← the HOOK, always open on face\n"
        "2. Then alternate: broll (12-20s) → face_cam (4-8s) → broll (12-20s) → face_cam (4-8s) ...\n"
        "3. Final segment: face_cam, 6-12s  ← the CTA close, always end on face\n"
        "4. broll segments: MINIMUM 12 seconds — never shorter, aim for 15-18s\n"
        "5. Middle face_cam cutbacks: 4-8 seconds ONLY — brief glimpse of the speaker\n"
        "6. Target 55-65% b-roll total coverage\n"
        f"7. All segments contiguous, covering 0.0 to {duration:.1f}s exactly, no gaps\n"
        "8. broll keyword: 2-4 concrete visual words for Pexels search\n\n"
        "EXAMPLE RHYTHM for a ~80s video:\n"
        "  0-10s face_cam (hook)\n"
        "  10-27s broll 'city coffee morning'\n"
        "  27-32s face_cam\n"
        "  32-50s broll 'laptop desk working'\n"
        "  50-55s face_cam\n"
        "  55-72s broll 'phone social media'\n"
        "  72-80s face_cam (CTA)\n\n"
        "Return ONLY a valid JSON array, no markdown, no explanation:\n"
        '[\n'
        '  {"start": 0.0, "end": 10.0, "type": "face_cam"},\n'
        '  {"start": 10.0, "end": 27.0, "type": "broll", "keyword": "city coffee morning"},\n'
        '  ...\n'
        f'  {{"start": X.X, "end": {duration:.1f}, "type": "face_cam"}}\n'
        "]\n"
        f"IMPORTANT: last segment MUST end at {duration:.1f} and MUST be face_cam."
    )

    text = _groq_llm(prompt, api_key)
    segments = json.loads(text)

    if segments:
        segments[-1]["end"] = duration

    merged = [segments[0]]
    for seg in segments[1:]:
        if seg["end"] - seg["start"] < 3.0:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def get_broll_keywords(transcript, api_key):
    text_sample = " ".join(s["text"] for s in transcript)[:1000]
    prompt = (
        "Given this video transcript, suggest 3 visual topics for b-roll footage.\n"
        f"TRANSCRIPT: {text_sample}\n\n"
        "Return ONLY a JSON array of 2-4 word Pexels search keywords:\n"
        '["keyword one", "keyword two", "keyword three"]'
    )
    result = _groq_llm(prompt, api_key)
    keywords = json.loads(result)
    return [k for k in keywords if isinstance(k, str)][:3]


def search_pexels_video(keyword):
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": keyword, "orientation": "portrait", "per_page": 10, "size": "small"}
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
    """Mix Edit mode: alternates face cam and b-roll segments."""
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

    # Free disk space — broll source clips no longer needed
    for clip in broll_clips.values():
        if clip and os.path.exists(clip):
            os.remove(clip)

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


def process_video_split_screen(video_path, job_id, api_key, progress_cb):
    """Split Screen mode: face cam on top half, b-roll looping on bottom half."""
    work = f"jobs/{job_id}"
    os.makedirs(work, exist_ok=True)

    progress_cb(3, "Reading video info…")
    width, height, duration = get_video_info(video_path)
    h2 = height // 2

    progress_cb(8, "Extracting audio…")
    audio_path = f"{work}/audio.mp3"
    extract_audio(video_path, audio_path)

    progress_cb(15, "Transcribing with Groq Whisper…")
    transcript = transcribe_audio(audio_path, api_key, duration)

    progress_cb(35, "Selecting b-roll themes…")
    keywords = get_broll_keywords(transcript, api_key)

    with open(f"{work}/edit_plan.json", "w") as _f:
        json.dump([{"type": "split_screen", "keywords": keywords}], _f)

    broll_paths = []
    for idx, kw in enumerate(keywords[:3]):
        pct = 45 + idx * 10
        progress_cb(pct, f'Downloading b-roll: "{kw}"…')
        url = search_pexels_video(kw)
        if url:
            clip = f"{work}/broll_{idx}.mp4"
            download_video(url, clip)
            broll_paths.append(clip)

    if not broll_paths:
        broll_paths = [video_path]

    progress_cb(75, "Building split-screen…")

    n = len(broll_paths)
    part_dur = duration / n
    bottom_parts = []
    for i, clip in enumerate(broll_paths):
        part_out = f"{work}/bottom_{i}.mp4"
        run_ffmpeg(
            "-stream_loop", "-1", "-i", clip,
            "-t", str(part_dur),
            "-vf", f"scale={width}:{h2}:force_original_aspect_ratio=increase,crop={width}:{h2}",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            part_out, error_label=f"bottom part {i}"
        )
        bottom_parts.append(part_out)

    if len(bottom_parts) == 1:
        bottom_track = bottom_parts[0]
    else:
        concat_txt = f"{work}/bottom_concat.txt"
        with open(concat_txt, "w") as f:
            for p in bottom_parts:
                f.write(f"file '{os.path.abspath(p)}'\n")
        bottom_track = f"{work}/bottom_track.mp4"
        run_ffmpeg(
            "-f", "concat", "-safe", "0", "-i", concat_txt,
            "-c:v", "copy",
            bottom_track, error_label="bottom concat"
        )

    top_track = f"{work}/top_track.mp4"
    run_ffmpeg(
        "-i", video_path,
        "-vf", f"scale={width}:{h2}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{h2}:(ow-iw)/2:(oh-ih)/2:black",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        top_track, error_label="top track"
    )

    for clip in broll_paths:
        if clip != video_path and os.path.exists(clip):
            os.remove(clip)

    progress_cb(92, "Compositing final video…")
    output = f"{work}/output.mp4"
    run_ffmpeg(
        "-i", top_track,
        "-i", bottom_track,
        "-i", video_path,
        "-filter_complex", "[0:v][1:v]vstack=inputs=2[out]",
        "-map", "[out]",
        "-map", "2:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output, error_label="split screen composite"
    )

    progress_cb(100, "Done!")
    return output
