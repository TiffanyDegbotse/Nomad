from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger
from core.utils import get_openai_client

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_OK = True
except Exception:
    CHROMA_OK = False

try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False


@dataclass
class MemoryRecord:
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]


class VectorStore:
    def __init__(self, settings: dict):
        self.settings = settings
        self.client = get_openai_client()
        self.emb_model = settings["openai"]["embedding_model"]
        self.use_chroma = bool(settings["rag"].get("use_chroma", True) and CHROMA_OK)
        self.collection_name = settings["rag"]["collection_name"]
        self.top_k = int(settings["rag"]["top_k"])
        self.min_score = float(settings["rag"].get("min_score", 0.0))

        if self.use_chroma:
            logger.info("VectorStore: using ChromaDB")
            db_path = "data/embeddings"
            self.chroma = chromadb.Client(ChromaSettings(persist_directory=db_path))
            self.coll = self.chroma.get_or_create_collection(
                self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        elif FAISS_OK:
            logger.info("VectorStore: using FAISS backend")
            self.faiss_index = None
            self.records: List[MemoryRecord] = []
        else:
            logger.warning("VectorStore: using in-memory fallback")
            self.records: List[MemoryRecord] = []

    # ---------------------------
    # Embedding helper
    # ---------------------------
    def _embed(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(model=self.emb_model, input=text)
        return emb.data[0].embedding

    # ---------------------------
    # Add a new memory
    # ---------------------------
    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        emb = self._embed(text)
        rec_id = metadata.get("id") or metadata.get("timestamp") or str(len(text))

        if self.use_chroma:
            self.coll.add(
                ids=[rec_id],
                embeddings=[emb],
                documents=[text],
                metadatas=[metadata],
            )
        else:
            rec = MemoryRecord(id=rec_id, text=text, metadata=metadata, embedding=emb)
            self.records.append(rec)
            if FAISS_OK:
                self._rebuild_faiss()

        return rec_id

    # ---------------------------
    # Rebuild FAISS index
    # ---------------------------
    def _rebuild_faiss(self):
        if not FAISS_OK:
            return

        vecs = np.array([r.embedding for r in self.records], dtype="float32")
        if vecs.size == 0:
            return

        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine if normalized
        faiss.normalize_L2(vecs)
        index.add(vecs)
        self.faiss_index = index

    # ---------------------------
    # Query similar memories
    # ---------------------------
    def query(self, text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        q = self._embed(text)

        if self.use_chroma:
            res = self.coll.query(
                query_embeddings=[q],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            out = []
            for i in range(len(res.get("ids", [[]])[0])):
                out.append({
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": 1 - res["distances"][0][i],
                })
            return out

        elif FAISS_OK and getattr(self, "faiss_index", None) is not None and len(self.records) > 0:
            vec = np.array([q], dtype="float32")
            faiss.normalize_L2(vec)
            D, I = self.faiss_index.search(vec, k)
            out = []
            for dist, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                rec = self.records[idx]
                out.append({
                    "text": rec.text,
                    "metadata": rec.metadata,
                    "score": float(dist),
                })
            return out

        else:
            def cos(a, b):
                a, b = np.array(a), np.array(b)
                return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

            scored = [
                {"text": r.text, "metadata": r.metadata, "score": cos(q, r.embedding)}
                for r in self.records
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:k]


__all__ = ["VectorStore"]
