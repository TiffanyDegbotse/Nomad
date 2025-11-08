# core/rag_qa.py
from __future__ import annotations
from typing import Dict, Any
from loguru import logger
import json
from core.utils import get_openai_client
from core.vector_store import VectorStore
from core.tts_engine import TTSEngine


class NomadQA:
    """
    Question-answering layer for Nomad.
    Retrieves memories, infers relationships between scenes,
    and generates TTS audio for the Raspberry Pi to play.
    """

    def __init__(self, settings: dict):
        self.settings = settings
        self.client = get_openai_client()
        self.vs = VectorStore(settings)
        self.tts = TTSEngine(settings)
        self.chat_model = settings["openai"]["chat_model"]
        self.temperature = settings["openai"].get("temperature", 0.4)

    def ask(self, question: str) -> Dict[str, Any]:
        logger.info(f"NomadQA received question: {question}")

        # 1️⃣ Retrieve memories
        retrieved = self.vs.query(question, top_k=5)
        if not retrieved:
            reply = "I don’t have any memories that match that yet."
            audio_path = self.tts.synth(reply, play_audio=False)
            logger.info(f"Audio saved for Pi: {audio_path}")
            return {"question": question, "reply": reply, "audio": str(audio_path)}

        # 2️⃣ Parse metadata & sort chronologically
        enriched = []
        for r in retrieved:
            ts = None
            try:
                meta = r.get("metadata", {})
                if "meta" in meta:
                    meta_obj = json.loads(meta["meta"])
                    ts = meta_obj.get("timestamp")
            except Exception:
                pass
            enriched.append({**r, "timestamp": ts})

        enriched.sort(key=lambda x: x.get("timestamp") or "", reverse=False)

        # 3️⃣ Boost location-bearing memories
        location_keywords = ["sign", "town", "city", "welcome", "landmark", "state", "road", "highway"]
        boosted = sorted(
            enriched,
            key=lambda x: any(k.lower() in x["text"].lower() for k in location_keywords),
            reverse=True
        )

        # 4️⃣ Build contextual memory text
        context_blocks = []
        for i, r in enumerate(boosted):
            context_blocks.append(
                f"[Memory {i+1} | {r.get('timestamp','unknown')}] {r['text']}"
            )

        context_text = "\n".join(context_blocks)

        # 5️⃣ Prompt for cross-memory reasoning
        prompt = f"""
You are Nomad, an AI travel companion who remembers past trips.
Here are your memory logs, listed from oldest to newest:

{context_text}

The user asked: "{question}"

Use your memories together. If a memory mentions a town, state, or signboard, 
assume it's part of the same trip as nearby memories.
Infer the user's likely current location or region.
Be confident, brief, and natural in your reply.
"""

        # 6️⃣ GPT reasoning
        try:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are Nomad, an AI co-pilot for road travel."},
                    {"role": "user", "content": prompt},
                ],
            )
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"NomadQA model error: {e}")
            reply = "Sorry, I couldn’t recall that right now."

        # 7️⃣ Generate TTS (for Pi)
        try:
            audio_path = self.tts.synth(reply, play_audio=False)
            logger.info(f"TTS saved to {audio_path} for Pi playback.")
        except Exception as e:
            logger.warning(f"TTS generation failed: {e}")
            audio_path = None

        return {
            "question": question,
            "reply": reply,
            "audio": str(audio_path) if audio_path else None,
        }


__all__ = ["NomadQA"]
