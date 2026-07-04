# Nomad

An AI-powered road trip companion that watches the road, remembers where you've been, and answers questions by voice. Nomad combines computer vision, retrieval-augmented memory, and speech interaction—optionally paired with a Raspberry Pi camera and speaker for a hands-free in-car experience.

## Features

- **Continuous vision** — Background polling of camera images detects road signs and landmarks via GPT-4o Vision
- **Journey memory (RAG)** — Detected locations are embedded and stored in ChromaDB for semantic recall during conversation
- **Voice interaction** — Wake word ("Nomad") triggers speech-to-text Q&A; responses are synthesized and played back through the Pi
- **Photo capture** — Say "take a picture" to trigger an on-demand capture from the Pi camera
- **Vintage postcards** — Transform trip photos into 1940s-style travel posters with DALL-E 3
- **Offline pipeline** — A simplified YOLO + OCR + ChatGPT path for single-image sign analysis without Pi hardware

## Project Structure

```
Nomad/
├── README.md
├── architecture.md
├── requirements.txt
├── .env                        # API keys (not committed)
├── nomad/                      # Python package
│   ├── config.py               # Shared paths and settings
│   ├── companion.py            # Voice + vision + RAG (main app)
│   ├── pipeline.py             # YOLO/OCR/ChatGPT batch pipeline
│   ├── postcard.py             # Vintage postcard generator
│   └── pi_client.py            # Pi image polling utility
└── data/                       # Runtime data (gitignored except samples)
    ├── images/
    │   ├── stream/             # Continuous camera frames
    │   └── captures/           # On-demand captures
    ├── audio/                  # TTS responses
    ├── db/                     # ChromaDB vector store
    ├── output/                 # Pipeline JSON results
    │   └── detected_signs/     # OCR crop outputs
    ├── postcards/              # Generated artwork + gallery
    └── samples/                # Test images and fixtures
```

See [architecture.md](architecture.md) for component diagrams, data flows, and design decisions.

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o, GPT-4o-mini, DALL-E 3, TTS)
- Microphone (for voice mode)
- Optional: Raspberry Pi with HTTP camera/audio server

### Python dependencies

```bash
pip install -r requirements.txt
```

Core packages: `openai`, `python-dotenv`, `requests`, `chromadb`, `sentence-transformers`, `speechrecognition`

Optional (batch pipeline): `opencv-python`, `numpy`, `ultralytics`, `easyocr`

## Setup

1. Clone the repository and create a virtual environment:

```bash
git clone <repo-url>
cd Nomad
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-key-here
PI_HOST=http://10.194.194.1:8080
PI_HOST_IP=10.194.194.1
```

3. Run commands from the project root so the `nomad` package resolves correctly.

## Usage

### Full voice companion

```bash
python -m nomad.companion
```

- Say **"Nomad"** followed by a question (e.g., "What signs have we passed recently?")
- Say **"take a picture"** to capture a photo from the Pi camera

### Simplified sign analysis (single image)

```bash
python -m nomad.pipeline data/samples/test1.jpg
```

Results are saved to `data/output/result_<timestamp>.json`.

### Postcard generator

```bash
python -m nomad.postcard
```

Uses the first `.jpg` in `data/samples/` by default. Generated postcards and a gallery HTML page are written to `data/postcards/`.

### Pi image polling

```bash
python -m nomad.pi_client
```

Polls the Pi every 60 seconds and saves frames to `data/images/stream/`. Press `Ctrl+C` to stop.

## Raspberry Pi Integration

Nomad expects a lightweight HTTP server on the Pi exposing:

| Endpoint     | Method | Purpose                          |
|--------------|--------|----------------------------------|
| `/photo.jpg` | GET    | Latest continuous camera frame   |
| `/cap.jpg`   | GET    | On-demand capture                |
| `/audio`     | POST   | Receive TTS audio for playback   |

## Environment Variables

| Variable         | Default                    | Description                |
|------------------|----------------------------|----------------------------|
| `OPENAI_API_KEY` | —                          | OpenAI API key (required)  |
| `PI_HOST`        | `http://10.194.194.1:8080` | Pi base URL for companion  |
| `PI_HOST_IP`     | `10.194.194.1`             | Pi IP for image polling    |
| `VISION_POLL_INTERVAL` | `10`                 | Seconds between vision polls |
| `IMAGE_POLL_INTERVAL`  | `60`                 | Seconds between image downloads |

## License

See repository license file for details.
