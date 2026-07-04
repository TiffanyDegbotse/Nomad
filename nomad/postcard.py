#!/usr/bin/env python3
"""Vintage postcard generator from user photos."""

import base64
from datetime import datetime
from pathlib import Path

import openai
import requests

from nomad import config


class PostcardGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key required!")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.postcards = []
        config.ensure_dirs()
        print("✅ Postcard Generator ready")

    def encode_image(self, image_path):
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    def create_postcard(self, photo_path, location_name):
        print(f"\n📮 Creating postcard for {location_name}...")
        print(f"   📸 Using photo: {Path(photo_path).name}")

        photo = Path(photo_path)
        if not photo.exists():
            print(f"   ❌ Photo not found: {photo_path}")
            return None

        base64_image = self.encode_image(photo_path)
        print("   🔍 Analyzing your photo...")

        analysis_prompt = f"""Analyze this photo taken at {location_name}.

Describe:
1. What's in the photo (buildings, nature, people, etc.)
2. The mood/atmosphere
3. Time of day
4. Colors and lighting
5. Any notable features

Keep it brief (2-3 sentences)."""

        try:
            analysis_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }],
                max_tokens=200,
            )
            photo_description = analysis_response.choices[0].message.content
            print(f"   📝 Photo contains: {photo_description[:100]}...")
        except Exception as e:
            print(f"   ⚠️  Could not analyze photo: {e}")
            photo_description = f"A scenic view of {location_name}"

        print("   🎨 Generating vintage postcard artwork (30 seconds)...")
        city_name = location_name.split(",")[0].strip()

        artwork_prompt = f"""Create a vintage 1940s travel poster for {location_name}.

Photo context: {photo_description}

Style requirements:
- Classic American travel poster aesthetic
- Warm autumn colors (oranges, browns, golds, burgundy)
- Art deco influence
- Painted illustration style (NOT photographic)
- Elegant composition with decorative ornate border
- Nostalgic, romantic feel
- Include vintage car from 1940s era
- Dramatic sky with sunset/sunrise
- Autumn foliage framing the scene

Text layout:
- Bottom section: Large bold text "{city_name.upper()}" in vintage serif typography
- Below that: "NORTH CAROLINA" in elegant letters
- Ornate decorative frame around entire poster

Incorporate elements from the photo description but make it artistic and stylized, not realistic.

This should look like a professional vintage WPA travel poster from the 1940s."""

        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=artwork_prompt,
                size="1024x1792",
                quality="standard",
                n=1,
            )

            image_data = requests.get(response.data[0].url).content
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = location_name.replace(",", "").replace(" ", "_")
            filename = f"postcard_{safe_name}_{timestamp}.png"
            postcard_path = config.POSTCARDS_DIR / filename
            postcard_path.write_bytes(image_data)

            print(f"   ✅ Postcard created: {filename}")

            postcard_info = {
                "location": location_name,
                "path": str(postcard_path),
                "filename": filename,
                "original_photo": str(photo_path),
                "timestamp": timestamp,
                "date": datetime.now().strftime("%B %d, %Y"),
                "description": photo_description,
            }
            self.postcards.append(postcard_info)
            return str(postcard_path)
        except Exception as e:
            print(f"   ❌ Error generating postcard: {e}")
            return None

    def generate_postcards_page(self):
        if not self.postcards:
            print("No postcards to display!")
            return None

        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Travel Postcards</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Georgia', serif;
            background: linear-gradient(135deg, #8B4513 0%, #D2691E 50%, #CD853F 100%);
            min-height: 100vh; padding: 40px 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        header { text-align: center; color: white; margin-bottom: 60px; padding: 20px; }
        h1 { font-size: 4.5em; margin-bottom: 15px; text-shadow: 4px 4px 8px rgba(0,0,0,0.6); }
        .subtitle { font-size: 1.5em; font-style: italic; opacity: 0.95; }
        .count { font-size: 1.2em; margin-top: 15px; background: rgba(255,255,255,0.2); display: inline-block; padding: 10px 25px; border-radius: 25px; }
        .postcards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 50px; padding: 30px 0; }
        .postcard-item { background: white; border-radius: 20px; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.4); cursor: pointer; position: relative; }
        .postcard-image { width: 100%; height: auto; display: block; }
        .postcard-info { padding: 30px; background: linear-gradient(to bottom, #f8f5f0 0%, #ede8e0 100%); border-top: 3px solid #D2691E; }
        .postcard-number { position: absolute; top: 20px; right: 20px; background: rgba(210, 105, 30, 0.95); color: white; padding: 8px 18px; border-radius: 25px; font-weight: bold; }
        .postcard-location { font-size: 1.8em; font-weight: bold; color: #8B4513; margin-bottom: 12px; }
        .postcard-date { font-size: 1em; color: #666; font-style: italic; margin-bottom: 15px; }
        .postcard-description { font-size: 0.95em; color: #555; line-height: 1.6; padding-top: 15px; border-top: 1px solid #ddd; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.95); justify-content: center; align-items: center; }
        .modal-content { max-width: 90%; max-height: 90vh; border-radius: 15px; }
        .close { position: absolute; top: 30px; right: 50px; color: white; font-size: 60px; font-weight: bold; cursor: pointer; }
        .download-btn { position: absolute; bottom: 30px; right: 50px; background: #D2691E; color: white; padding: 15px 30px; border-radius: 30px; text-decoration: none; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📮 MY POSTCARDS</h1>
            <p class="subtitle">Vintage Memories from My Journey</p>
            <p class="count">🎨 """ + f"{len(self.postcards)} postcard{'s' if len(self.postcards) != 1 else ''} collected" + """</p>
        </header>
        <div class="postcards-grid">
"""

        for idx, postcard in enumerate(reversed(self.postcards)):
            html += f"""
            <div class="postcard-item" onclick="openModal('{postcard['path']}')">
                <span class="postcard-number">#{len(self.postcards) - idx}</span>
                <img src="{postcard['path']}" alt="{postcard['location']}" class="postcard-image">
                <div class="postcard-info">
                    <div class="postcard-location">{postcard['location']}</div>
                    <div class="postcard-date">📅 {postcard['date']}</div>
                    <div class="postcard-description">"{postcard['description'][:150]}..."</div>
                </div>
            </div>
"""

        html += """
        </div>
    </div>
    <div id="myModal" class="modal" onclick="closeModal(event)">
        <span class="close" onclick="closeModal(event)">&times;</span>
        <img class="modal-content" id="modalImg">
        <a href="#" id="downloadLink" class="download-btn" download>⬇️ Download Postcard</a>
    </div>
    <script>
        function openModal(imagePath) {
            event.stopPropagation();
            document.getElementById('modalImg').src = imagePath;
            document.getElementById('downloadLink').href = imagePath;
            document.getElementById('myModal').style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
        function closeModal(event) {
            event.stopPropagation();
            document.getElementById('myModal').style.display = 'none';
            document.body.style.overflow = 'auto';
        }
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') closeModal(event);
        });
    </script>
</body>
</html>
"""

        output_path = config.POSTCARDS_DIR / "my_postcards.html"
        output_path.write_text(html, encoding="utf-8")
        print(f"\n✅ Postcards page updated: {output_path}")
        print(f"   📮 Total postcards: {len(self.postcards)}")
        return str(output_path)


def main():
    print("=" * 70)
    print("📮 POSTCARD GENERATOR - Using Your Photos!")
    print("=" * 70)

    generator = PostcardGenerator()

    sample_photo = config.SAMPLES_DIR / "test1.jpg"
    if not sample_photo.exists():
        sample_photo = next(config.SAMPLES_DIR.glob("*.jpg"), None)

    if sample_photo is None:
        print("❌ No sample photo found in data/samples/. Add a .jpg file and re-run.")
        return

    postcard = generator.create_postcard(
        photo_path=str(sample_photo),
        location_name="Durham, North Carolina",
    )

    if postcard:
        page_path = generator.generate_postcards_page()
        print(f"\n🎉 SUCCESS!")
        print(f"📂 Open in browser: {page_path}")


if __name__ == "__main__":
    main()
