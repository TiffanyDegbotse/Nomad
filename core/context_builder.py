from __future__ import annotations
import json
import pathlib
import io, base64
from typing import Dict, Any, List
from loguru import logger
from PIL import Image
from core.utils import ROOT, now_iso, b64_image, read_text, get_openai_client


class ContextBuilder:
    """
    Handles scene understanding for the car assistant.
    Given road sign or scene images, it uses GPT-4o (multimodal)
    to extract relevant text, landmarks, and contextual clues.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self.client = get_openai_client()
        self.system_prompt = read_text(ROOT / "config" / "prompts" / "system_prompt.txt")
        self.vision_prompt = read_text(ROOT / "config" / "prompts" / "vision_prompt.txt")
        self.chat_model = settings["openai"]["chat_model"]
        self.temperature = settings["openai"].get("temperature", 0.3)

    def analyze_images(self, image_paths: List[pathlib.Path]) -> Dict[str, Any]:
        """Send image(s) to OpenAI Vision (GPT-4o) and return structured output."""
        if not image_paths:
            return {
                "sign_text": None,
                "landmark": None,
                "location_hint": None,
                "confidence": 0.0,
                "notes": "no image",
            }

        # Build the user message content
        user_content = [{"type": "text", "text": self.vision_prompt}]
        for path in image_paths:
            try:
                # Always convert to valid JPEG before encoding (fixes invalid format issues)
                buf = io.BytesIO()
                Image.open(path).convert("RGB").save(buf, format="JPEG")
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to attach image {path}: {e}")

        logger.info(f"Analyzing {len(image_paths)} image(s) with {self.chat_model}...")

        try:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as e:
            logger.error(f"OpenAI Vision API call failed: {e}")
            return {
                "sign_text": None,
                "landmark": None,
                "location_hint": None,
                "confidence": 0.0,
                "notes": f"API error: {e}",
            }

        text = resp.choices[0].message.content or "{}"
        logger.info(f"Vision model raw output: {text[:200]}")

        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {
                "sign_text": None,
                "landmark": None,
                "location_hint": text[:200],
                "confidence": 0.5,
                "notes": "unstructured",
            }

        parsed["timestamp"] = now_iso()
        return parsed

    def summarize_context(self, vision_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the vision output into a semantic unit for storage."""
        sign_text = vision_dict.get("sign_text")
        landmark = vision_dict.get("landmark")
        hint = vision_dict.get("location_hint")

        summary = ", ".join([x for x in [sign_text, landmark, hint] if x]) or "Unclear scene"
        return {
            "summary": summary,
            "tags": [t for t in ["sign", "landmark", "history"] if t],
            # Convert nested metadata dict to JSON string to satisfy Chroma
            "meta": json.dumps(vision_dict, ensure_ascii=False),
        }


__all__ = ["ContextBuilder"]
