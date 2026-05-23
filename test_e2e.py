"""End-to-end test — runs the full pipeline on test_input.mp4"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

def _load_api_key():
    # Standard env var
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    # Managed Claude Code environment — session ingress token
    tf = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE", "")
    if tf and os.path.exists(tf):
        with open(tf) as f:
            return f.read().strip()
    return ""

API_KEY = _load_api_key()
if not API_KEY:
    print("ERROR: No Anthropic API key found. Set ANTHROPIC_API_KEY.")
    sys.exit(1)

INPUT = "test_input.mp4"
if not os.path.exists(INPUT):
    print(f"ERROR: {INPUT} not found — run make_test_video.sh first")
    sys.exit(1)

os.makedirs("jobs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

log = []
def progress_cb(pct, msg):
    line = f"[{pct:3d}%] {msg}"
    print(line)
    log.append(line)

print("=" * 55)
print("  B-Roll Editor — End-to-End Test")
print("=" * 55)

from processor import process_video

job_id = "test_" + str(int(time.time()))
try:
    output = process_video(INPUT, job_id, API_KEY, progress_cb)
    print("\n" + "=" * 55)
    print(f"  SUCCESS! Output: {output}")
    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")

    # Quick sanity check with ffprobe
    import subprocess, json as _json
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", output],
        capture_output=True, text=True, check=True
    )
    info = _json.loads(probe.stdout)
    streams = info.get("streams", [])
    vid = next((s for s in streams if s["codec_type"] == "video"), None)
    aud = next((s for s in streams if s["codec_type"] == "audio"), None)
    fmt = info.get("format", {})
    dur = float(fmt.get("duration", 0))
    print(f"  Duration:   {dur:.1f}s")
    print(f"  Video:      {vid['width']}x{vid['height']} @ {vid.get('codec_name','?')}")
    print(f"  Audio:      {aud.get('codec_name','?')} / {aud.get('sample_rate','?')} Hz")
    print("=" * 55)

except Exception as e:
    import traceback
    print("\nFAILED:")
    traceback.print_exc()
    sys.exit(1)
