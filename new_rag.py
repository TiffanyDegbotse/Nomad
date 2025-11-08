#!/usr/bin/env python3
"""
AI Journey Reconstruction - Pure OpenAI Vision Pipeline
+ Conversational Memory Layer (Nomad Chat)
"""

import os
import base64
import json
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv  # 👈 NEW
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PIL import Image as PILImage
from PIL.ExifTags import TAGS
from loguru import logger
from openai import OpenAI

# --- Load .env file ---
load_dotenv()


# ===============================================================
# IMAGE ANALYSIS
# ===============================================================

class OpenAIImageProcessor:
    """Process images using OpenAI Vision API only"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key required! Set OPENAI_API_KEY environment variable")
        self.client = OpenAI(api_key=self.api_key)
        print("✅ OpenAI Vision client initialized")

    def encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def detect_and_extract_text(self, image_path):
        """Use OpenAI Vision to detect signs and extract text"""
        print(f"   🔍 Analyzing with OpenAI Vision...")

        base64_image = self.encode_image(image_path)
        prompt = """Analyze this image and identify any road, town, or location signs.
Return JSON only:
{"signs":[{"text":"exact text","sign_type":"welcome/town/highway","confidence":0.95}]}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            detections = []
            if "signs" in result:
                for sign in result["signs"]:
                    if sign.get("text"):
                        detections.append({
                            "text": sign["text"],
                            "confidence": sign.get("confidence", 0.8),
                            "sign_type": sign.get("sign_type", "unknown"),
                            "method": "openai_vision"
                        })
                        print(f"      📝 '{sign['text']}' ({sign.get('sign_type')}, "
                              f"conf: {sign.get('confidence', 0.8):.2f})")

            if not detections:
                print(f"      ℹ️  No location signs detected")

            return {
                "image_path": str(image_path),
                "has_signs": len(detections) > 0,
                "detections": detections,
            }

        except json.JSONDecodeError:
            print("⚠️  Failed to parse response as JSON")
            return {"image_path": str(image_path), "has_signs": False, "detections": []}
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return {"image_path": str(image_path), "has_signs": False, "detections": [], "error": str(e)}


# ===============================================================
# DATABASE MANAGEMENT
# ===============================================================

class JourneyDatabase:
    """Manage journey data in vector database"""

    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model

    def add_location(self, location_text, timestamp, image_id,
                     confidence, has_sign=True, additional_metadata=None):
        """Add detected location to vector database"""
        embedding = self.embedding_model.encode(location_text).tolist()
        metadata = {
            "timestamp": timestamp,
            "image_id": str(image_id),
            "location_text": location_text,
            "confidence": float(confidence),
            "has_sign": has_sign,
            "detection_type": "sign_detected" if has_sign else "inferred",
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        clean_timestamp = timestamp.replace(":", "").replace("-", "").replace("T", "_")
        doc_id = f"loc_{image_id}_{clean_timestamp}"

        try:
            self.collection.add(
                embeddings=[embedding],
                documents=[location_text],
                metadatas=[metadata],
                ids=[doc_id],
            )
            print("      ✅ Added to vector DB")
            return doc_id
        except Exception as e:
            print(f"⚠️  DB error: {e}")
            return None


# ===============================================================
# UTILITIES
# ===============================================================

def get_image_timestamp(image_path):
    """Extract timestamp from EXIF or filename"""
    try:
        img = PILImage.open(image_path)
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ["DateTime", "DateTimeOriginal"]:
                    dt_str = str(value).replace(":", "-", 2)
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    return dt.isoformat()
    except Exception:
        pass

    filename = Path(image_path).stem
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        dt = datetime.strptime(f"{match[1]}{match[2]}", "%Y%m%d%H%M%S")
        return dt.isoformat()

    try:
        return datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat()
    except Exception:
        return datetime.now().isoformat()


def process_images_folder(folder_path, processor, database):
    """Process all images in a folder"""
    folder = Path(folder_path)
    image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not image_files:
        print(f"⚠️  No images found in {folder_path}")
        return []
    results = []
    print(f"\n📂 Found {len(image_files)} images in {folder_path}")
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] {image_path.name}")
        ts = get_image_timestamp(image_path)
        print(f"   ⏰ Timestamp: {ts}")
        result = processor.detect_and_extract_text(image_path)
        if result["has_signs"]:
            for det in result["detections"]:
                database.add_location(
                    location_text=det["text"],
                    timestamp=ts,
                    image_id=idx,
                    confidence=det["confidence"],
                    has_sign=True,
                    additional_metadata={
                        "image_name": image_path.name,
                        "method": "openai_vision",
                        "sign_type": det.get("sign_type", "unknown"),
                    },
                )
        results.append({
            "image_id": idx,
            "image_name": image_path.name,
            "timestamp": ts,
            "has_signs": result["has_signs"],
            "num_detections": len(result["detections"]),
            "detections": result["detections"],
        })
    return results


def display_summary(results):
    """Display summary"""
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    total = len(results)
    with_signs = sum(1 for r in results if r["has_signs"])
    detections = sum(r["num_detections"] for r in results)
    print(f"\n📊 Total: {total} images | With signs: {with_signs} | Detections: {detections}")
    print(f"\n📍 Locations:")
    unique = set()
    for r in results:
        if r["has_signs"]:
            for d in r["detections"]:
                unique.add(d["text"])
                print(f"   • {r['timestamp'][:19]} - {d['text']} (conf: {d['confidence']:.2f})")
    print(f"\n🗺️  Unique locations: {len(unique)}")
    for loc in sorted(unique):
        print(f"   • {loc}")


# CONVERSATIONAL LAYER
class JourneyChatAssistant:
    """Conversational layer for querying stored travel memories"""

    def __init__(self, db_path="./journey_db", collection_name="journey_locations", model="gpt-4o-mini"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.db_client.get_or_create_collection(name=collection_name)
        self.model = model

        print("💬 Journey Chat Assistant initialized")

    def query_memories(self, question, top_k=5):
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = embed_model.encode(question).tolist()
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        memories = []
        for i, doc in enumerate(res["documents"][0]):
            memories.append({
                "text": doc,
                "metadata": res["metadatas"][0][i],
                "score": 1 - res["distances"][0][i],
            })
        return memories

    def ask(self, question):
        logger.info(f"🧭 Question: {question}")
        memories = self.query_memories(question)
        if not memories:
            print("Nomad: I don’t have any memories matching that yet.")
            return
        context = "\n".join(
            [f"- {m['text']} (conf: {m['metadata'].get('confidence', 0):.2f}, time: {m['metadata'].get('timestamp', 'unknown')})"
             for m in memories]
        )
        prompt = f"""
You are Nomad, an AI travel companion.
Use these past travel memories to answer naturally.

Question: {question}
Context:
{context}

Respond as a friendly travel guide in 2-3 sentences.
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are Nomad, an AI travel companion with memory."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
            )
            reply = resp.choices[0].message.content.strip()
            print(f"\n💬 Nomad: {reply}")
        except Exception as e:
            print(f"⚠️  Chat model error: {e}")


# ===============================================================
# MAIN PIPELINE
# ===============================================================

def main():
    print("=" * 70)
    print("🚗 AI JOURNEY RECONSTRUCTION + NOMAD CHAT")
    print("=" * 70)

    IMAGES_FOLDER = "./test_images"
    if not Path(IMAGES_FOLDER).exists():
        Path(IMAGES_FOLDER).mkdir(parents=True, exist_ok=True)
        print(f"📸 Add your images to {IMAGES_FOLDER} and run again.")
        return

    print("\n📦 Step 1: Initializing vector database...")
    client = chromadb.PersistentClient(path="./journey_db")
    try:
        collection = client.get_collection(name="journey_locations")
        print("   ℹ️  Using existing collection")
    except:
        collection = client.create_collection(name="journey_locations")
        print("   ℹ️  Created new collection")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    database = JourneyDatabase(collection, embedding_model)
    print("   ✅ Database ready")

    print("\n📦 Step 2: Initializing OpenAI Vision processor...")
    try:
        processor = OpenAIImageProcessor()
    except ValueError as e:
        print(f"   ❌ {e}")
        return

    print(f"\n📸 Step 3: Processing images from {IMAGES_FOLDER}")
    results = process_images_folder(IMAGES_FOLDER, processor, database)
    if not results:
        print("⚠️  No images processed.")
        return

    display_summary(results)
    with open("processing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Results saved to processing_results.json")

    # ========================
    # 🗣️  Conversational Chat
    # ========================
    assistant = JourneyChatAssistant()
    while True:
        q = input("\nAsk Nomad a question (or 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        assistant.ask(q)


if __name__ == "__main__":
    main()
