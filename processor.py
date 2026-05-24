import os
import json
import random
import subprocess
import requests

PEXELS_API_KEY = "LppRQMMFSN0E7avfYQUoeQVdifahsxZwB0Uzag6Z7OQhZvemwAYfQ5eu"
GROQ_BASE = "https://api.groq.com/openai/v1"
MAX_WIDTH = 720  # cap to avoid OOM on 512 MB containers


def run_ffmpeg(*args, error_label="ffmpeg"):
    result = subprocess.run(
        ["ffmpeg", "-y", "-threads", "1"] + list(args),
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


def _cap_resolution(width, height):
    """Cap to MAX_WIDTH, keep aspect ratio, ensure dims divisible by 4."""
    if width > MAX_WIDTH:
        height = int(height * MAX_WIDTH / width)
        width = MAX_WIDTH
    width -= width % 2
    height -= height % 4  # divisible by 4 so split-screen half is always even
    return width, height


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


def _sentences_in_range(transcript, start, end):
    """Return transcript sentences that overlap [start, end], clamped to that range."""
    result = []
    for s in transcript:
        s_start = max(s["start"], start)
        s_end = min(s["end"], end)
        if s_end - s_start >= 0.5:
            result.append({"start": s_start, "end": s_end, "text": s["text"]})
    return result


def analyze_segments(transcript, duration, api_key):
    lines = "\n".join(
        f"[{s['start']:.1f}s-{s['end']:.1f}s]: {s['text']}"
        for s in transcript
    )

    n_broll = 4 if duration >= 120 else 3

    prompt = (
        "You are a professional video editor creating a beautiful, balanced short-form edit.\n\n"
        f"TRANSCRIPT ({duration:.0f}s):\n{lines}\n\n"
        f"Create exactly {n_broll} b-roll segments using this EXACT rhythm:\n"
        "  1. face_cam — HOOK: cover the first 1-2 sentences (6-10s)\n"
        f"  2. Alternate {n_broll} times: broll (12-20s) then face_cam (4-6s)\n"
        "  3. The LAST face_cam is the CTA — MUST be 5-10s, covering the final 1-2 sentences\n\n"
        f"EXAMPLE for a {duration:.0f}s video with {n_broll} b-roll segments:\n"
        "  0-8s    face_cam  ← hook, first sentence\n"
        "  8-25s   broll\n"
        "  25-30s  face_cam  ← 5s cutback\n"
        "  30-47s  broll\n"
        "  47-52s  face_cam  ← 5s cutback\n"
        "  52-69s  broll\n"
        f"  69-{duration:.0f}s  face_cam  ← CTA, MINIMUM 5s, last sentence(s)\n\n"
        "Return ONLY valid JSON, no markdown, no keywords:\n"
        '[\n'
        '  {"start": 0.0, "end": 8.0, "type": "face_cam"},\n'
        '  {"start": 8.0, "end": 25.0, "type": "broll"},\n'
        '  {"start": 25.0, "end": 30.0, "type": "face_cam"},\n'
        '  ...\n'
        f'  {{"start": X.X, "end": {duration:.1f}, "type": "face_cam"}}\n'
        "]\n"
        f"RULES: exactly {n_broll} broll entries. "
        f"First and last segments are face_cam. Last face_cam is 5-10s. Last ends at {duration:.1f}."
    )

    text = _groq_llm(prompt, api_key)
    segments = json.loads(text)

    if segments:
        segments[-1]["end"] = duration

    # Hard guarantee: last segment must be face_cam and at least 5s
    if segments and segments[-1]["type"] != "face_cam":
        segments.append({"start": duration - 7.0, "end": duration, "type": "face_cam"})
        segments[-2]["end"] = duration - 7.0
    if segments and segments[-1]["type"] == "face_cam":
        cta_dur = segments[-1]["end"] - segments[-1]["start"]
        if cta_dur < 5.0 and len(segments) >= 2:
            deficit = 5.0 - cta_dur
            segments[-2]["end"] -= deficit
            segments[-1]["start"] -= deficit

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


def search_pexels_video(query):
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "orientation": "portrait", "per_page": 15}
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers=headers, params=params, timeout=30
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if not videos:
        return None

    random.shuffle(videos)  # fresh clip every render

    for video in videos:
        for vf in sorted(video.get("video_files", []),
                         key=lambda x: x.get("height", 0), reverse=True):
            if vf.get("width", 9999) < vf.get("height", 0):
                return vf["link"]

    files = videos[0].get("video_files", [])
    return files[0]["link"] if files else None


def download_video(url, dest):
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)


def cut_face_cam(original, start, dur, out, w, h):
    run_ffmpeg(
        "-ss", str(start), "-i", original,
        "-vf", f"setpts=PTS-STARTPTS,"
               f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black",
        "-r", "30",
        "-frames:v", str(round(dur * 30)),
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        out, error_label="face-cam cut"
    )


def cut_broll(broll_path, dur, out, w, h):
    run_ffmpeg(
        "-stream_loop", "-1", "-i", broll_path,
        "-vf", f"setpts=PTS-STARTPTS,"
               f"scale={w}:{h}:force_original_aspect_ratio=increase,"
               f"crop={w}:{h}",
        "-r", "30",
        "-frames:v", str(round(dur * 30)),
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        out, error_label="b-roll cut"
    )


def process_video(video_path, job_id, api_key, progress_cb):
    """Mix Edit mode: alternates face cam and b-roll segments."""
    work = f"jobs/{job_id}"
    os.makedirs(work, exist_ok=True)

    progress_cb(3, "Reading video info…")
    width, height, duration = get_video_info(video_path)
    width, height = _cap_resolution(width, height)

    progress_cb(8, "Extracting audio…")
    audio_path = f"{work}/audio.mp3"
    extract_audio(video_path, audio_path)

    progress_cb(15, "Transcribing with Groq Whisper…")
    transcript = transcribe_audio(audio_path, api_key, duration)

    progress_cb(40, "Planning b-roll segments with Groq LLaMA…")
    segments = analyze_segments(transcript, duration, api_key)

    with open(f"{work}/edit_plan.json", "w") as _f:
        json.dump(segments, _f)

    seg_files = []
    n_segs = len(segments)

    for i, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        out = f"{work}/seg_{i:03d}.mp4"
        pct = 50 + int((i / n_segs) * 35)

        if seg["type"] == "face_cam":
            progress_cb(pct, f"Face cam {i + 1}/{n_segs} ({dur:.0f}s)…")
            cut_face_cam(video_path, seg["start"], dur, out, width, height)
        else:
            # One b-roll clip per sentence spoken during this segment
            sentences = _sentences_in_range(transcript, seg["start"], seg["end"])
            if not sentences:
                sentences = [{"start": seg["start"], "end": seg["end"], "text": ""}]

            total_sent = sum(s["end"] - s["start"] for s in sentences)
            sent_clips = []

            for j, sent in enumerate(sentences):
                raw = sent["end"] - sent["start"]
                clip_dur = max(raw * dur / total_sent, 1.0) if total_sent > 0 else max(dur / len(sentences), 1.0)
                sc_out = f"{work}/bsc_{i}_{j}.mp4"
                query = sent["text"].strip()

                url = None
                if query:
                    progress_cb(pct, f'B-roll: "{query[:50]}"…')
                    url = search_pexels_video(query)
                    if not url and len(query.split()) > 3:
                        url = search_pexels_video(" ".join(query.split()[:4]))

                if url:
                    tmp = f"{work}/btmp_{i}_{j}.mp4"
                    download_video(url, tmp)
                    cut_broll(tmp, clip_dur, sc_out, width, height)
                    os.remove(tmp)
                else:
                    cut_face_cam(video_path, sent["start"], clip_dur, sc_out, width, height)

                sent_clips.append(sc_out)

            cat_txt = f"{work}/bcat_{i}.txt"
            with open(cat_txt, "w") as f:
                for sc in sent_clips:
                    f.write(f"file '{os.path.abspath(sc)}'\n")
            run_ffmpeg(
                "-f", "concat", "-safe", "0", "-i", cat_txt,
                "-vf", "setpts=PTS-STARTPTS",
                "-r", "30",
                "-frames:v", str(round(dur * 30)),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                out, error_label=f"broll concat {i}"
            )
            for sc in sent_clips:
                if os.path.exists(sc):
                    os.remove(sc)

        seg_files.append(out)

    progress_cb(87, "Joining clips…")
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

    progress_cb(95, "Adding original audio…")
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
    """Split Screen mode: face cam fills top half, per-sentence b-roll on bottom half."""
    work = f"jobs/{job_id}"
    os.makedirs(work, exist_ok=True)

    progress_cb(3, "Reading video info…")
    width, height, duration = get_video_info(video_path)
    width, height = _cap_resolution(width, height)
    h2 = height // 2
    total_frames = round(duration * 30)

    progress_cb(8, "Extracting audio…")
    audio_path = f"{work}/audio.mp3"
    extract_audio(video_path, audio_path)

    progress_cb(15, "Transcribing with Groq Whisper…")
    transcript = transcribe_audio(audio_path, api_key, duration)

    # One b-roll clip per sentence — same approach as mix edit
    sentences = _sentences_in_range(transcript, 0, duration)
    if not sentences:
        sentences = [{"start": 0.0, "end": duration, "text": ""}]

    display_kws = [s["text"][:40] for s in sentences[:5]]
    with open(f"{work}/edit_plan.json", "w") as _f:
        json.dump([{"type": "split_screen", "keywords": display_kws}], _f)

    total_sent = sum(s["end"] - s["start"] for s in sentences)
    sent_clips = []
    n_sent = len(sentences)

    for j, sent in enumerate(sentences):
        pct = 35 + int((j / n_sent) * 45)
        raw = sent["end"] - sent["start"]
        clip_dur = max(raw * duration / total_sent, 1.0) if total_sent > 0 else max(duration / n_sent, 1.0)
        query = sent["text"].strip()
        sc_out = f"{work}/bsc_{j}.mp4"

        url = None
        if query:
            progress_cb(pct, f'B-roll: "{query[:50]}"…')
            url = search_pexels_video(query)
            if not url and len(query.split()) > 3:
                url = search_pexels_video(" ".join(query.split()[:4]))

        if url:
            tmp = f"{work}/btmp_{j}.mp4"
            download_video(url, tmp)
            cut_broll(tmp, clip_dur, sc_out, width, h2)
            os.remove(tmp)
        else:
            # Fallback: use original video cropped to fill bottom half
            run_ffmpeg(
                "-ss", str(sent["start"]), "-i", video_path,
                "-vf", f"setpts=PTS-STARTPTS,"
                       f"scale={width}:{h2}:force_original_aspect_ratio=increase,"
                       f"crop={width}:{h2}",
                "-r", "30",
                "-frames:v", str(round(clip_dur * 30)),
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                sc_out, error_label=f"fallback bottom {j}"
            )

        sent_clips.append(sc_out)

    progress_cb(82, "Building split-screen…")

    if len(sent_clips) == 1:
        bottom_track = sent_clips[0]
    else:
        cat_txt = f"{work}/bottom_concat.txt"
        with open(cat_txt, "w") as f:
            for sc in sent_clips:
                f.write(f"file '{os.path.abspath(sc)}'\n")
        bottom_track = f"{work}/bottom_track.mp4"
        run_ffmpeg(
            "-f", "concat", "-safe", "0", "-i", cat_txt,
            "-vf", "setpts=PTS-STARTPTS",
            "-r", "30",
            "-frames:v", str(total_frames),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            bottom_track, error_label="bottom concat"
        )
        for sc in sent_clips:
            if os.path.exists(sc):
                os.remove(sc)

    # Top: crop to fill entire width — face stays visible, sides may be cropped
    top_track = f"{work}/top_track.mp4"
    run_ffmpeg(
        "-i", video_path,
        "-vf", f"setpts=PTS-STARTPTS,"
               f"scale={width}:{h2}:force_original_aspect_ratio=increase,"
               f"crop={width}:{h2}",
        "-r", "30",
        "-frames:v", str(total_frames),
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        top_track, error_label="top track"
    )

    progress_cb(92, "Compositing final video…")
    # Step 1: vstack top + bottom into a video-only file (no audio)
    video_only = f"{work}/video_only.mp4"
    run_ffmpeg(
        "-i", top_track,
        "-i", bottom_track,
        "-filter_complex", "[0:v][1:v]vstack=inputs=2[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        video_only, error_label="vstack"
    )

    # Step 2: mux original audio with -c:v copy (no re-encode = no encoder delay = perfect sync)
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
