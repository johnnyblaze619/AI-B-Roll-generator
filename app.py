import os
import uuid
import threading
import json
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload limit

os.makedirs("uploads", exist_ok=True)
os.makedirs("jobs", exist_ok=True)

jobs = {}  # job_id -> {progress, message, status, output}


def _env_api_key() -> str:
    """Return the Groq API key from the environment."""
    return os.environ.get("GROQ_API_KEY", "")


@app.route("/")
def index():
    has_env_key = bool(_env_api_key())
    return render_template("index.html", has_env_key=has_env_key)


@app.route("/process", methods=["POST"])
def process():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "No video uploaded"}), 400

    api_key = request.form.get("api_key") or _env_api_key()
    if not api_key:
        return jsonify({"error": "Groq API key is required"}), 400

    job_id = str(uuid.uuid4())
    upload_path = f"uploads/{job_id}.mp4"
    video.save(upload_path)

    jobs[job_id] = {
        "progress": 0,
        "message": "Queued…",
        "status": "running",
        "output": None,
        "edit_plan": None,
    }

    def run():
        try:
            from processor import process_video

            def progress_cb(pct, msg):
                jobs[job_id]["progress"] = pct
                jobs[job_id]["message"] = msg

            output = process_video(upload_path, job_id, api_key, progress_cb)
            jobs[job_id]["status"] = "done"
            jobs[job_id]["output"] = output
        except Exception as exc:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = str(exc)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404

    if job.get("edit_plan") is None:
        plan_path = f"jobs/{job_id}/edit_plan.json"
        if os.path.exists(plan_path):
            with open(plan_path) as f:
                job["edit_plan"] = json.load(f)

    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return "Not ready", 404
    return send_file(
        os.path.abspath(job["output"]),
        as_attachment=True,
        download_name="broll_output.mp4",
        mimetype="video/mp4",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
