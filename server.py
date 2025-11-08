from flask import Flask, send_file
import os
from pathlib import Path

app = Flask(__name__)

AUDIO_PATH = Path("data/transcripts/tts_output.mp3")

@app.route("/audio/latest")
def serve_audio():
    if AUDIO_PATH.exists():
        return send_file(AUDIO_PATH, mimetype="audio/mpeg")
    else:
        return {"error": "No audio file yet."}, 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
