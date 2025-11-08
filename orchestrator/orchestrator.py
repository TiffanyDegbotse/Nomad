from __future__ import annotations
import pathlib
import time
from loguru import logger
from core.capture_handler import CaptureHandler
from core.utils import ROOT, read_yaml
from orchestrator.handlers import PipelineHandlers


class Orchestrator:
    """
    Main controller for the multimodal car assistant.
    Handles image capture, vision analysis, context summarization,
    vector storage, and text-to-speech playback.
    """

    def __init__(self):
        # Load configuration
        self.settings = read_yaml(ROOT / "config" / "settings.yaml")
        self.handlers = PipelineHandlers(self.settings)

        # Polling interval (seconds)
        self.loop_interval = int(self.settings["orchestrator"].get("loop_interval_seconds", 5))

        # Directory where road-sign images are placed (e.g., from Raspberry Pi)
        self.capture = CaptureHandler(images_dir=ROOT / self.settings["paths"]["images_dir"])

    def run_once(self):
        """
        Run one full cycle:
        1. Get new road-sign or scene image(s)
        2. Analyze via GPT-4o Vision
        3. Store summary in vector DB
        4. Generate speech output
        """
        images = self.capture.get_new_images(limit=1)
        if not images:
            logger.info("No new images found. Drop a .jpg/.png into data/images to simulate.")
            return

        result = self.handlers.process_images(images)
        if not result:
            logger.warning("No result returned from pipeline.")
            return

        # --- Log structured outputs ---
        summary = result.get("summary", "No summary available")
        audio_path = result.get("audio")

        logger.info(f"RAG Summary: {summary}")
        if audio_path:
            logger.info(f"Audio saved: {audio_path}")
        else:
            logger.warning("No audio path returned from TTS.")

    def run_loop(self):
        """
        Continuously run the pipeline — ideal for Raspberry Pi deployment.
        Watches the image folder and processes new inputs every few seconds.
        """
        logger.info("Starting orchestrator loop… (Ctrl+C to stop)")
        while True:
            try:
                self.run_once()
                time.sleep(self.loop_interval)
            except KeyboardInterrupt:
                logger.info("Orchestrator stopped manually.")
                break
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}")
                time.sleep(self.loop_interval)
