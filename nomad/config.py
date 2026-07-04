"""Shared paths and environment configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

IMAGES_STREAM_DIR = DATA_DIR / "images" / "stream"
IMAGES_CAPTURE_DIR = DATA_DIR / "images" / "captures"
AUDIO_DIR = DATA_DIR / "audio"
DB_DIR = DATA_DIR / "db"
OUTPUT_DIR = DATA_DIR / "output"
DETECTED_SIGNS_DIR = OUTPUT_DIR / "detected_signs"
POSTCARDS_DIR = DATA_DIR / "postcards"
SAMPLES_DIR = DATA_DIR / "samples"

PI_HOST = os.getenv("PI_HOST", "http://10.194.194.1:8080")
PI_HOST_IP = os.getenv("PI_HOST_IP", "10.194.194.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
SIGN_CONFIDENCE_THRESHOLD = float(os.getenv("SIGN_CONFIDENCE_THRESHOLD", "0.3"))
VISION_POLL_INTERVAL = int(os.getenv("VISION_POLL_INTERVAL", "10"))
IMAGE_POLL_INTERVAL = int(os.getenv("IMAGE_POLL_INTERVAL", "60"))

ALL_DIRS = (
    IMAGES_STREAM_DIR,
    IMAGES_CAPTURE_DIR,
    AUDIO_DIR,
    DB_DIR,
    OUTPUT_DIR,
    DETECTED_SIGNS_DIR,
    POSTCARDS_DIR,
    SAMPLES_DIR,
)


def ensure_dirs() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
