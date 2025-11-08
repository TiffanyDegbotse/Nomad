from __future__ import annotations
import pathlib
from typing import Optional
from loguru import logger
from playsound import playsound
from core.utils import get_openai_client, ROOT


class TTSEngine:
    """
    Handles text-to-speech synthesis using OpenAI TTS models,
    and plays or saves the resulting MP3 file without relying on PyDub.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self.client = get_openai_client()
        self.model = settings["openai"]["tts_model"]
        self.voice = settings["openai"].get("tts_voice", "alloy")

    def synth(
        self,
        text: str,
        save_to: Optional[pathlib.Path] = None,
        play_audio: bool = False
    ) -> Optional[pathlib.Path]:
        """Generate and optionally play MP3 speech for given text."""
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3",  # correct parameter name
            )
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

        # Some SDK versions return `response.content`; fallback to .read()
        data = getattr(response, "content", None)
        if data is None:
            try:
                data = response.read()
            except Exception:
                logger.error("No audio content in TTS response.")
                return None

        # Save output
        out_path = save_to or (ROOT / "data" / "transcripts" / "tts_output.mp3")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        logger.info(f"TTS audio saved to: {out_path}")

        # Optional playback
        if play_audio:
            try:
                playsound(str(out_path))
            except Exception as e:
                logger.warning(f"Audio playback failed: {e}")

        return out_path
