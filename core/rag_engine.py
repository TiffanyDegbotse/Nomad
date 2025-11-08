from __future__ import annotations
from typing import Dict, Any, List
from .utils import get_openai_client, read_text, ROOT


class RAGEngine:
    def __init__(self, settings: dict, vector_store):
        self.settings = settings
        self.client = get_openai_client()
        self.vector_store = vector_store
        self.system_prompt = read_text(ROOT / "config" / "prompts" / "system_prompt.txt")
        self.chat_model = settings["openai"]["chat_model"]
        self.temperature = settings["openai"].get("temperature", 0.3)

    def respond(self, current_context: Dict[str, Any]) -> str:
        # Retrieve memory
        query_text = current_context.get("summary") or "recent sights"
        memories = self.vector_store.query(query_text)

        memory_bullets = []
        for m in memories:
            memory_bullets.append(
                f"- {m['text']} (seen: {m['metadata'].get('meta', {}).get('timestamp', 'unknown')})"
            )
        memory_str = "\n".join(memory_bullets[:5]) if memory_bullets else "(no prior matches)"

        user_prompt = (
            "Context for driver (latest):\n"
            f"{current_context.get('summary')}\n\n"
            "Related memories from earlier in the trip:\n"
            f"{memory_str}\n\n"
            "Give a short, speakable historical/cultural note that connects "
            "the current sight to any related prior sights."
        )

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return resp.choices[0].message.content.strip()
