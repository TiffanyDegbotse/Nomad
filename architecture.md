# Nomad Architecture

This document describes the system design, component interactions, and data flows for the Nomad AI road trip companion.

## Overview

Nomad is built around two operational modes:

1. **Full companion** (`nomad.companion`) — A long-running, voice-driven system with continuous vision, persistent journey memory, and Raspberry Pi I/O.
2. **Batch analysis** (`nomad.pipeline`) — A single-image pipeline for sign detection and AI recommendations without hardware dependencies.

Additional utilities handle image polling (`nomad.pi_client`) and souvenir postcard generation (`nomad.postcard`).

All paths and settings are centralized in `nomad.config`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NOMAD FULL COMPANION                            │
│                                                                         │
│  ┌──────────────┐    HTTP GET     ┌─────────────────┐                  │
│  │ Raspberry Pi │◄───────────────│ ContinuousImage  │                  │
│  │  Camera Srv  │  /photo.jpg    │   Processor     │                  │
│  └──────┬───────┘  /cap.jpg      │ (bg thread)     │                  │
│         │                          └────────┬─────────┘                  │
│         │ POST /audio                       │ GPT-4o Vision               │
│         ▼                                   ▼                             │
│  ┌──────────────┐                  ┌─────────────────┐                  │
│  │   Speaker    │                  │  Sign Detection  │                  │
│  └──────────────┘                  └────────┬─────────┘                  │
│                                            │                               │
│                                            ▼                               │
│                                   ┌─────────────────┐                  │
│                                   │  ChromaDB RAG   │                  │
│                                   │   (data/db)     │                  │
│                                   └────────┬─────────┘                  │
│                                            │                               │
│  ┌──────────────┐   wake word    ┌─────────▼────────┐                  │
│  │  Microphone  │───────────────►│   Voice Loop     │                  │
│  └──────────────┘   "Nomad"      │  (main thread)   │                  │
│                                  └────────┬─────────┘                  │
│                                           │                               │
│                              query ───────┤                               │
│                              reply ───────┤                               │
│                              TTS ─────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Repository Layout

```
nomad/                  Application code
├── config.py           Paths, env vars, directory bootstrap
├── companion.py        Voice + vision + RAG main loop
├── pipeline.py         YOLO/OCR/ChatGPT batch pipeline
├── postcard.py         DALL-E postcard generator
└── pi_client.py        Pi image polling

data/                   Runtime artifacts (mostly gitignored)
├── images/stream/      Continuous camera frames
├── images/captures/    On-demand captures
├── audio/              TTS output
├── db/                 ChromaDB persistent store
├── output/             Pipeline JSON + OCR crops
├── postcards/          Generated posters + HTML gallery
└── samples/            Committed test fixtures
```

## Core Components

### 1. Configuration (`nomad.config`)

Single source of truth for:

- Project root and all `data/` subdirectories
- OpenAI and Pi connection settings from environment variables
- `ensure_dirs()` called at startup by each module

### 2. Continuous Image Processor

**File:** `nomad.companion` — `ContinuousImageProcessor`

A daemon background thread that runs on a configurable poll interval (default 10s).

| Step | Action |
|------|--------|
| Fetch | `GET {PI_HOST}/photo.jpg` → save to `data/images/stream/` |
| Analyze | Send image to GPT-4o Vision with a JSON-only sign detection prompt |
| Store | Embed detected text with `all-MiniLM-L6-v2` and upsert into ChromaDB |

The processor deduplicates by filename (`last_fetched` set) so the same frame is not processed twice.

On-demand capture uses `GET {PI_HOST}/cap.jpg` and saves to `data/images/captures/`, triggered by the voice command "take a picture".

### 3. Journey Memory (RAG)

**Storage:** ChromaDB persistent client at `data/db/`, collection `journey_locations`

Each detected sign becomes a memory record:

| Field | Source |
|-------|--------|
| `documents` | Sign/landmark text |
| `embeddings` | MiniLM vector of the text |
| `metadatas` | `timestamp`, `image_name`, `confidence` |
| `ids` | `mem_<unix_timestamp>` |

**Query path:** When the user asks a question, the question is embedded with the same model and ChromaDB returns the top-5 nearest memories by cosine distance. These are injected as context into a GPT-4o-mini completion.

### 4. Voice Interface

**File:** `nomad.companion` — main loop

Runs on the foreground thread while the image processor runs in the background.

| Stage | Technology |
|-------|------------|
| Speech-to-text | Google Speech Recognition via `speech_recognition` |
| Text-to-speech | OpenAI `gpt-4o-mini-tts`, voice `alloy` |
| Audio delivery | `POST {PI_HOST}/audio` with multipart file upload |

### 5. Simplified Road Trip Pipeline

**File:** `nomad.pipeline` — `RoadTripPipeline`

A three-stage batch pipeline for offline or development use:

```
Image → SignDetector → TextExtractor → AIRecommendationEngine → data/output/
         YOLOv8n or CV    EasyOCR           GPT-4o
```

**SignDetector** tries YOLOv8 first (`yolov8n.pt`, confidence ≥ 0.3). If YOLO is unavailable or finds nothing, it falls back to OpenCV contour detection for rectangular regions.

**TextExtractor** crops each bounding box, runs EasyOCR, and also attempts full-image OCR as a fallback. Cropped regions are saved to `data/output/detected_signs/`.

Each optional dependency degrades gracefully with console warnings.

### 6. Postcard Generator

**File:** `nomad.postcard` — `PostcardGenerator`

1. **Analyze** — GPT-4o Vision describes the user's photo
2. **Generate** — DALL-E 3 creates a 1024×1792 vintage 1940s travel poster

Output is saved to `data/postcards/` with a self-contained HTML gallery.

### 7. Image Polling Utility

**File:** `nomad.pi_client`

Polls `http://{PI_HOST_IP}:8080/test.jpg` on a configurable interval (default 60s) and saves frames to `data/images/stream/`.

## Data Stores

| Path | Type | Contents |
|------|------|----------|
| `data/db/` | ChromaDB | Embedded sign/landmark memories |
| `data/images/stream/` | Filesystem | Continuous camera frames |
| `data/images/captures/` | Filesystem | On-demand captures |
| `data/audio/` | Filesystem | Latest TTS response |
| `data/postcards/` | Filesystem | Generated poster PNGs + HTML gallery |
| `data/output/` | Filesystem | Pipeline JSON results and sign crops |
| `data/samples/` | Filesystem | Test images and fixture JSON |

## External Services

| Service | Models Used | Purpose |
|---------|-------------|---------|
| OpenAI Chat | `gpt-4o`, `gpt-4o-mini` | Vision analysis, Q&A, recommendations |
| OpenAI Images | `dall-e-3` | Postcard artwork |
| OpenAI Audio | `gpt-4o-mini-tts` | Voice responses |
| Google Speech API | — | Wake word and question transcription |
| Hugging Face (local) | `all-MiniLM-L6-v2` | Text embeddings for RAG |

## Threading Model

`nomad.companion` uses a single-process, dual-thread design:

| Thread | Role | Blocking? |
|--------|------|-----------|
| Main | Voice wake-word loop, STT, RAG query, TTS | Blocks on microphone I/O |
| Background | Pi polling, vision, memory writes | Daemon; dies with main process |

ChromaDB handles concurrent read/write at the collection level; no explicit locking is used.

## Configuration

| Setting | Source | Default |
|---------|--------|---------|
| `OPENAI_API_KEY` | `.env` | — |
| `PI_HOST` | `.env` | `http://10.194.194.1:8080` |
| `PI_HOST_IP` | `.env` | `10.194.194.1` |
| `VISION_POLL_INTERVAL` | `.env` | `10` |
| `IMAGE_POLL_INTERVAL` | `.env` | `60` |
| Embedding model | code | `all-MiniLM-L6-v2` |
| YOLO model | `.env` | `yolov8n.pt` |
| RAG top-k | code | `5` |

## Error Handling Strategy

All components follow a **degrade gracefully** pattern:

- Pi unreachable → log warning, skip frame, retry on next poll
- Vision JSON parse failure → treat as no signs detected
- Missing optional dependency → skip that stage, use fallback or simulated output
- STT failure → prompt user and continue listening
- TTS/Pi audio POST failure → log warning; reply is still printed to console

## Future Considerations

- **Unified CLI** — Single entry point with subcommands (`companion`, `pipeline`, `postcard`, `poll`)
- **Proximity-aware RAG** — Weight memories by recency and geographic relevance
- **On-device vision** — Replace GPT-4o Vision polling with local YOLO/OCR to reduce API cost and latency
