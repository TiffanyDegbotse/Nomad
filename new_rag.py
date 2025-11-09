#!/usr/bin/env python3
"""
NOMAD Voice Edition (Fast Start)
Continuous Vision Context + Wake Word Interaction
Optimized for speed and responsiveness
"""

import os
import time
import json
import base64
import threading
import requests
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import chromadb
from loguru import logger

# ================================================================
# SETUP
# ================================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PI_HOST = "http://10.194.194.1:8080"
IMAGES_DIR = Path("./test_images")
AUDIO_DIR = Path("./data/audio")
CAP_DIR = Path("./cap_images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CAP_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting NOMAD — initializing components...")

# Load the embedding model once (this is what used to cause lag)
print("🧠 Loading embedding model (MiniLM)...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

print("💾 Connecting to vector database...")
chroma_client = chromadb.PersistentClient(path="./journey_db")
collection = chroma_client.get_or_create_collection(name="journey_locations")

print("✅ Initialization complete! NOMAD is starting up.\n")

# ================================================================
# VISION CONTEXT BUILDER (BACKGROUND THREAD)
# ================================================================
class ContinuousImageProcessor(threading.Thread):
    def __init__(self, collection):
        super().__init__(daemon=True)
        self.collection = collection
        self.embedding_model = EMBED_MODEL
        self.last_fetched = set()

    def run(self):
        print("📸 Background image processor running...")
        while True:
            try:
                image_path = self.fetch_image()
                if image_path and image_path.name not in self.last_fetched:
                    self.last_fetched.add(image_path.name)
                    result = self.analyze_image(image_path)
                    if result["has_signs"]:
                        for det in result["detections"]:
                            self.add_to_memory(det["text"], image_path, det)
            except Exception as e:
                print(f"⚠️ Background loop error: {e}")
            time.sleep(10)  # less frequent, frees CPU

    def fetch_image(self):
        """Pull current image from Pi"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = IMAGES_DIR / f"photo_{timestamp}.jpg"
        try:
            resp = requests.get(f"{PI_HOST}/photo.jpg", timeout=5)
            if resp.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"🖼  New image: {filename.name}")
                return filename
            else:
                print(f"⚠️ Pi returned {resp.status_code}")
        except Exception as e:
            print(f"⚠️ Could not fetch image: {e}")
        return None
    
    def capture_image(self):
        """Pull current image from Pi"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = CAP_DIR / f"photo_{timestamp}.jpg"
        try:
            resp = requests.get(f"{PI_HOST}/cap.jpg", timeout=5)
            if resp.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"🖼  New image: {filename.name}")
                return filename
            else:
                print(f"Pi returned {resp.status_code}")
        except Exception as e:
            print(f"Could not fetch image: {e}")
        return None



    def analyze_image(self, image_path):
        """Use GPT-4o Vision to detect signs"""
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        prompt = """Analyze this image and detect any visible location, sign, or landmark.
Respond ONLY in JSON:
{"signs":[{"text":"exact text or landmark","confidence":0.9}]}"""

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "high"}}
                    ]
                }],
                max_tokens=300,
                temperature=0.2,
            )

            text = resp.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            data = json.loads(text)
        except Exception as e:
            print(f"⚠️ Vision parse error: {e}")
            data = {"signs": []}

        detections = data.get("signs", [])
        return {"has_signs": bool(detections), "detections": detections}

    def add_to_memory(self, text, image_path, detection):
        emb = self.embedding_model.encode(text).tolist()
        meta = {
            "timestamp": datetime.now().isoformat(),
            "image_name": image_path.name,
            "confidence": detection.get("confidence", 0.8)
        }
        self.collection.add(
            embeddings=[emb],
            documents=[text],
            metadatas=[meta],
            ids=[f"mem_{int(time.time())}"]
        )
        print(f"🧠 Added memory: '{text}' ({meta['confidence']:.2f})")


# ================================================================
# VOICE INTERFACE (FOREGROUND LOOP)
# ================================================================
def calibrate_mic():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎚️  Calibrating microphone for background noise...")
        r.adjust_for_ambient_noise(source, duration=1)
    print("🎤 Microphone ready.")
    return r


def wait_for_wake_word(r):
    with sr.Microphone() as source:
        print("\n🎧 Waiting for 'Nomad'...")
        audio = r.listen(source, timeout=None, phrase_time_limit=4)
    try:
        text = r.recognize_google(audio).lower()
        if "nomad" in text:
            print("👂 Wake word detected!")
            return True
    except Exception:
        pass
    return False

def wait_for_wake_word_picture(r):
    with sr.Microphone() as source:
        print("\n🎧 Waiting for 'take a picture'...")
        audio = r.listen(source, timeout=None, phrase_time_limit=4)
    try:
        text = r.recognize_google(audio).lower()
        print(f"🗣️ You said: {text}")
        if "take a picture" in text:
            print("👂 Wake word detected!")
            return True
    except Exception:
        pass
    return False



def capture_question(r):
    with sr.Microphone() as source:
        print("🎙️ Ask your question...")
        audio = r.listen(source, timeout=8, phrase_time_limit=10)
    try:
        question = r.recognize_google(audio)
        print(f"🗣️ You said: {question}")
        return question
    except Exception:
        print("⚠️ Didn't catch that.")
        return None


# ================================================================
# MEMORY QUERY + RESPONSE
# ================================================================
def query_memories(collection, question, top_k=5):
    q_emb = EMBED_MODEL.encode(question).tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    memories = []
    for i, doc in enumerate(res["documents"][0]):
        memories.append({
            "text": doc,
            "metadata": res["metadatas"][0][i],
            "score": 1 - res["distances"][0][i],
        })
    return memories


def generate_reply(question, memories):
    if not memories:
        context = "(no memories yet)"
    else:
        context = "\n".join(
            [f"- {m['text']} (conf: {m['metadata'].get('confidence', 0):.2f}, time: {m['metadata'].get('timestamp', 'unknown')})"
             for m in memories]
        )
    prompt = f"""
You are Nomad, an AI travel companion.
Use these memories to answer naturally.

Question: {question}
Context:
{context}

Respond as a friendly travel guide in 2–3 sentences.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Nomad, an AI travel companion."},
            {"role": "user", "content": prompt},
        ]
    )
    reply = resp.choices[0].message.content.strip()
    print(f"💬 Nomad: {reply}")
    return reply


def synthesize_and_send(reply):
    """Generate TTS and send to Pi HTTP server"""
    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply
        )
        audio_path = AUDIO_DIR / "nomad_reply.mp3"
        with open(audio_path, "wb") as f:
            f.write(speech.read())
        print(f"💾 Audio saved: {audio_path}")

        # Send to Pi for playback
        with open(audio_path, "rb") as f:
            res = requests.post(f"{PI_HOST}/audio", files={"file": f}, timeout=10)
            if res.status_code == 200:
                print("📡 Sent to Pi for playback")
            else:
                print(f"⚠️ Pi response: {res.status_code}")
    except Exception as e:
        print(f"⚠️ Could not send audio: {e}")


# ================================================================
# MAIN LOOP
# ================================================================
def main():
    print("=" * 70)
    print("🚗 NOMAD — Fast Start Edition")
    print("=" * 70)

    # Start background vision thread
    bg = ContinuousImageProcessor(collection)
    bg.start()

    # Prepare microphone
    recognizer = calibrate_mic()

    # Main voice loop
    while True:
        if wait_for_wake_word(recognizer):
            q = capture_question(recognizer)
            if q:
                memories = query_memories(collection, q)
                reply = generate_reply(q, memories)
                synthesize_and_send(reply)
        time.sleep(0.5)
        if wait_for_wake_word_picture(recognizer):
            bg.capture_image()




if __name__ == "__main__":
    main()
