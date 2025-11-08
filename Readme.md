# ---------------------------------
# SightSense – Multimodal RAG Car Assistant


A Raspberry Pi–camera workflow where each roadside image is sent to OpenAI Vision to infer **sign text / landmarks / location hints**, converted into a compact **context summary**, stored in a **vector database** (Chroma/FAISS), then combined with prior trip memory (RAG) to generate a **speakable historical/cultural narration** via TTS.


## Quickstart


1. **Clone & install**
```bash
python -m venv .venv && source .venv/bin/activate # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```


2. **Add keys**
- Create `config/keys.env` with `OPENAI_API_KEY=...`


3. **Adjust settings (optional)**
- Edit `config/settings.yaml` to change models, voice, or RAG params.


4. **Drop a sample image**
- Put a .jpg/.png road-sign or landmark photo in `data/images/`.


5. **Run**
```bash
python main.py
# or continuous loop:
# python -c "from orchestrator.orchestrator import Orchestrator as O; O().run_loop()"
```


6. **(Optional) Run API**
```bash
uvicorn orchestrator.routes:app --reload --port 8080
# POST an image from the Pi
# curl -F "file=@data/images/sample.jpg" http://localhost:8080/ingest
```


## Notes
- The Pi team can either: (A) save images into `data/images/` (shared volume) or (B) POST to `/ingest`.
- If Chroma is unavailable, FAISS or a simple in-memory fallback is used automatically.
- Audio playback is saved as MP3 in `data/transcripts/tts_output.mp3`. Use your OS player or wire to the car’s speakers.


## Safety
- Keep responses short for driver attention. Do not show maps/directions while moving.
- Respect privacy; do not store raw faces. Persist only compact context summaries.

