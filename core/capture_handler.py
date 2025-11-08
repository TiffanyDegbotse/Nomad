from __future__ import annotations
import pathlib
from typing import Optional, List
from loguru import logger
from core.utils import ROOT


class CaptureHandler:
    """
    Simulates or manages image capture for the car camera.
    For now, it just checks the /data/images folder for existing image files.
    On the Raspberry Pi, this will later integrate with the Pi Camera.
    """

    def __init__(self, images_dir: Optional[pathlib.Path] = None):
        self.images_dir = images_dir or (ROOT / "data" / "images")
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def get_new_images(self, limit: int = 1) -> List[pathlib.Path]:
        """
        Return a list of image file paths (jpg/png) found in data/images.
        If no images are found, returns an empty list.
        """
        images = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        if not images:
            logger.info("No images found in data/images directory.")
        else:
            logger.info(f"Found {len(images)} image(s): {[p.name for p in images[:limit]]}")
        return images[:limit]
#for pi team