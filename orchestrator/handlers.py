from __future__ import annotations
import json
from loguru import logger
from core.context_builder import ContextBuilder
from core.vector_store import VectorStore
from core.tts_engine import TTSEngine
from core.utils import gen_id


class PipelineHandlers:
    """Coordinates the Vision → Context → Memory → Speech pipeline."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.ctx = ContextBuilder(settings)
        self.vs = VectorStore(settings)
        self.tts = TTSEngine(settings)

    def process_images(self, image_paths):
        if not image_paths:
            logger.warning("No new images to process.")
            return None

        # --- Step 1: Vision + reasoning ---
        vision_dict = self.ctx.analyze_images(image_paths)
        context = self.ctx.summarize_context(vision_dict)

        # --- Step 2: Add unique ID & store in memory ---
        ctx_id = gen_id("ctx")

        # context["meta"] is a JSON string — safe to store directly in Chroma
        metadata = {
            "id": ctx_id,
            "meta": context["meta"],  # keep as string
        }

        try:
            self.vs.add(context["summary"], metadata=metadata)
            logger.info(f"Context stored in VectorStore as {ctx_id}")
        except Exception as e:
            logger.warning(f"VectorStore add failed: {e}")

        # --- Step 3: Generate speech ---
        try:
            text = (
                f"Here's what I saw: {context['summary']}. "
                f"I've stored it in memory for later reference."
            )
            out_path = self.tts.synth(text, play_audio=True)
            logger.info(f"Speech generated at {out_path}")
        except Exception as e:
            logger.warning(f"TTS failed: {e}")

        return context
