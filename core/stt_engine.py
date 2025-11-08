from __future__ import annotations
from .utils import get_openai_client


class STTEngine:
    def __init__(self, settings: dict):
        self.settings = settings
        self.client = get_openai_client()
        # You could integrate Whisper here if desired
