from __future__ import annotations
from typing import List, Dict, Any
from collections import deque


class ConversationManager:
    def __init__(self, max_turns: int = 10):
        self.buffer = deque(maxlen=max_turns)

    def add_turn(self, user: str, assistant: str, meta: Dict[str, Any]):
        self.buffer.append({
            "user": user,
            "assistant": assistant,
            "meta": meta
        })

    def recent(self) -> List[Dict[str, Any]]:
        return list(self.buffer)
