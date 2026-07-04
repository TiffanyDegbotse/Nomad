#!/usr/bin/env python3
"""NOMAD voice companion — continuous vision, RAG memory, and wake-word interaction."""

import base64
import json
import threading
import time
from datetime import datetime

import chromadb
import requests
import speech_recognition as sr
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from nomad import config

config.ensure_dirs()

client = OpenAI(api_key=config.OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=str(config.DB_DIR))
collection = chroma_client.get_or_create_collection(name="journey_locations")


class ContinuousImageProcessor(threading.Thread):
    def __init__(self, memory_collection):
        super().__init__(daemon=True)
        self.collection = memory_collection
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
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
            time.sleep(config.VISION_POLL_INTERVAL)

    def fetch_image(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = config.IMAGES_STREAM_DIR / f"photo_{timestamp}.jpg"
        try:
            resp = requests.get(f"{config.PI_HOST}/photo.jpg", timeout=5)
            if resp.status_code == 200:
                filename.write_bytes(resp.content)
                print(f"🖼  New image: {filename.name}")
                return filename
            print(f"⚠️ Pi returned {resp.status_code}")
        except Exception as e:
            print(f"⚠️ Could not fetch image: {e}")
        return None

    def capture_image(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = config.IMAGES_CAPTURE_DIR / f"photo_{timestamp}.jpg"
        try:
            resp = requests.get(f"{config.PI_HOST}/cap.jpg", timeout=5)
            if resp.status_code == 200:
                filename.write_bytes(resp.content)
                print(f"🖼  New image: {filename.name}")
                return filename
            print(f"Pi returned {resp.status_code}")
        except Exception as e:
            print(f"Could not fetch image: {e}")
        return None

    def analyze_image(self, image_path):
        b64_img = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        prompt = (
            "Analyze this image and detect any visible location, sign, or landmark.\n"
            'Respond ONLY in JSON:\n'
            '{"signs":[{"text":"exact text or landmark","confidence":0.9}]}'
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img}",
                                "detail": "high",
                            },
                        },
                    ],
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
            "confidence": detection.get("confidence", 0.8),
        }
        self.collection.add(
            embeddings=[emb],
            documents=[text],
            metadatas=[meta],
            ids=[f"mem_{int(time.time())}"],
        )
        print(f"🧠 Added memory: '{text}' ({meta['confidence']:.2f})")


def calibrate_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎚️  Calibrating microphone for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("🎤 Microphone ready.")
    return recognizer


def wait_for_wake_word(recognizer):
    with sr.Microphone() as source:
        print("\n🎧 Waiting for 'Nomad'...")
        audio = recognizer.listen(source, timeout=None, phrase_time_limit=4)
    try:
        text = recognizer.recognize_google(audio).lower()
        if "nomad" in text:
            print("👂 Wake word detected!")
            return True
    except Exception:
        pass
    return False


def wait_for_wake_word_picture(recognizer):
    with sr.Microphone() as source:
        print("\n🎧 Waiting for 'take a picture'...")
        audio = recognizer.listen(source, timeout=None, phrase_time_limit=4)
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"🗣️ You said: {text}")
        if "take a picture" in text:
            print("👂 Wake word detected!")
            return True
    except Exception:
        pass
    return False


def capture_question(recognizer):
    with sr.Microphone() as source:
        print("🎙️ Ask your question...")
        audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
    try:
        question = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {question}")
        return question
    except Exception:
        print("⚠️ Didn't catch that.")
        return None


def query_memories(embed_model, question, top_k=5):
    q_emb = embed_model.encode(question).tolist()
    res = collection.query(
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


def generate_reply(question, memories):
    if not memories:
        context = "(no memories yet)"
    else:
        context = "\n".join(
            f"- {m['text']} (conf: {m['metadata'].get('confidence', 0):.2f}, "
            f"time: {m['metadata'].get('timestamp', 'unknown')})"
            for m in memories
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
        ],
    )
    reply = resp.choices[0].message.content.strip()
    print(f"💬 Nomad: {reply}")
    return reply


def synthesize_and_send(reply):
    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply,
        )
        audio_path = config.AUDIO_DIR / "nomad_reply.mp3"
        audio_path.write_bytes(speech.read())
        print(f"💾 Audio saved: {audio_path}")

        with open(audio_path, "rb") as audio_file:
            res = requests.post(
                f"{config.PI_HOST}/audio",
                files={"file": audio_file},
                timeout=10,
            )
            if res.status_code == 200:
                print("📡 Sent to Pi for playback")
            else:
                print(f"⚠️ Pi response: {res.status_code}")
    except Exception as e:
        print(f"⚠️ Could not send audio: {e}")


def main():
    print("=" * 70)
    print("🚗 NOMAD — Fast Start Edition")
    print("=" * 70)

    print("🚀 Starting NOMAD — initializing components...")
    print("🧠 Loading embedding model (MiniLM)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("💾 Connecting to vector database...")
    print("✅ Initialization complete! NOMAD is starting up.\n")

    bg = ContinuousImageProcessor(collection)
    bg.start()

    recognizer = calibrate_mic()

    while True:
        if wait_for_wake_word(recognizer):
            question = capture_question(recognizer)
            if question:
                memories = query_memories(embed_model, question)
                reply = generate_reply(question, memories)
                synthesize_and_send(reply)
        time.sleep(0.5)
        if wait_for_wake_word_picture(recognizer):
            bg.capture_image()


if __name__ == "__main__":
    main()
