#!/usr/bin/env python3
"""
Postcard Generator - Uses User's Photo + Location
Creates vintage-style postcard from user's actual photo
"""

import openai
import os
import base64
import requests
from pathlib import Path
from datetime import datetime
import json


class PostcardGenerator:
    """Generate postcards using user's actual photos"""
    
    def __init__(self, api_key=None):
        self.api_key = ""
        if not self.api_key:
            raise ValueError("OpenAI API key required!")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.postcards = []
        
        # Create postcards directory
        Path("./postcards").mkdir(exist_ok=True)
        
        print("✅ Postcard Generator ready")
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def create_postcard(self, photo_path, location_name):
        """
        Create a vintage postcard using user's actual photo
        
        Args:
            photo_path: Path to user's photo
            location_name: Location name (e.g., "Durham, North Carolina")
        
        Returns:
            Path to generated postcard
        """
        
        print(f"\n📮 Creating postcard for {location_name}...")
        print(f"   📸 Using photo: {Path(photo_path).name}")
        
        # Check if photo exists
        if not Path(photo_path).exists():
            print(f"   ❌ Photo not found: {photo_path}")
            return None
        
        # Encode user's photo
        base64_image = self.encode_image(photo_path)
        
        # Step 1: Analyze the photo first
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
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            photo_description = analysis_response.choices[0].message.content
            print(f"   📝 Photo contains: {photo_description[:100]}...")
            
        except Exception as e:
            print(f"   ⚠️  Could not analyze photo: {e}")
            photo_description = f"A scenic view of {location_name}"
        
        # Step 2: Generate vintage postcard artwork INSPIRED by the photo
        print("   🎨 Generating vintage postcard artwork (30 seconds)...")
        
        # Extract just the city name for the prompt
        city_name = location_name.split(',')[0].strip()
        
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
                size="1024x1792",  # Portrait orientation
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            # Download the generated postcard
            image_data = requests.get(image_url).content
            
            # Save postcard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = location_name.replace(',', '').replace(' ', '_')
            filename = f"postcard_{safe_name}_{timestamp}.png"
            postcard_path = f"./postcards/{filename}"
            
            with open(postcard_path, 'wb') as f:
                f.write(image_data)
            
            print(f"   ✅ Postcard created: {filename}")
            
            # Store postcard info
            postcard_info = {
                'location': location_name,
                'path': postcard_path,
                'filename': filename,
                'original_photo': photo_path,
                'timestamp': timestamp,
                'date': datetime.now().strftime("%B %d, %Y"),
                'description': photo_description
            }
            
            self.postcards.append(postcard_info)
            
            return postcard_path
            
        except Exception as e:
            print(f"   ❌ Error generating postcard: {e}")
            return None
    
    def generate_postcards_page(self):
        """
        Generate beautiful "My Postcards" HTML page
        """
        
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
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Georgia', serif;
            background: linear-gradient(135deg, #8B4513 0%, #D2691E 50%, #CD853F 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            margin-bottom: 60px;
            padding: 20px;
        }
        
        h1 {
            font-size: 4.5em;
            margin-bottom: 15px;
            text-shadow: 4px 4px 8px rgba(0,0,0,0.6);
            letter-spacing: 3px;
            font-weight: bold;
        }
        
        .subtitle {
            font-size: 1.5em;
            font-style: italic;
            opacity: 0.95;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
        }
        
        .count {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 15px;
            background: rgba(255,255,255,0.2);
            display: inline-block;
            padding: 10px 25px;
            border-radius: 25px;
            backdrop-filter: blur(10px);
        }
        
        .postcards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 50px;
            padding: 30px 0;
        }
        
        .postcard-item {
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            cursor: pointer;
            position: relative;
        }
        
        .postcard-item:hover {
            transform: translateY(-20px) rotate(2deg) scale(1.02);
            box-shadow: 0 30px 60px rgba(0,0,0,0.5);
        }
        
        .postcard-image {
            width: 100%;
            height: auto;
            display: block;
            background: linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%);
        }
        
        .postcard-info {
            padding: 30px;
            background: linear-gradient(to bottom, #f8f5f0 0%, #ede8e0 100%);
            border-top: 3px solid #D2691E;
        }
        
        .postcard-number {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(210, 105, 30, 0.95);
            color: white;
            padding: 8px 18px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }
        
        .postcard-location {
            font-size: 1.8em;
            font-weight: bold;
            color: #8B4513;
            margin-bottom: 12px;
            line-height: 1.3;
        }
        
        .postcard-date {
            font-size: 1em;
            color: #666;
            font-style: italic;
            margin-bottom: 15px;
        }
        
        .postcard-description {
            font-size: 0.95em;
            color: #555;
            line-height: 1.6;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }
        
        /* Modal for full-size view */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.95);
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .modal-content {
            max-width: 90%;
            max-height: 90vh;
            border-radius: 15px;
            box-shadow: 0 0 50px rgba(255,255,255,0.2);
        }
        
        .close {
            position: absolute;
            top: 30px;
            right: 50px;
            color: white;
            font-size: 60px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .close:hover {
            color: #D2691E;
            transform: scale(1.1);
        }
        
        .download-btn {
            position: absolute;
            bottom: 30px;
            right: 50px;
            background: #D2691E;
            color: white;
            padding: 15px 30px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.1em;
            transition: 0.3s;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .download-btn:hover {
            background: #8B4513;
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(50px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        .postcard-item {
            animation: fadeInUp 0.6s ease-out backwards;
        }
        
        @media (max-width: 768px) {
            h1 {
                font-size: 2.8em;
            }
            
            .postcards-grid {
                grid-template-columns: 1fr;
                gap: 35px;
            }
            
            .close {
                font-size: 40px;
                top: 15px;
                right: 25px;
            }
            
            .download-btn {
                bottom: 15px;
                right: 25px;
                padding: 12px 20px;
                font-size: 0.95em;
            }
        }
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
        
        # Add postcards (newest first)
        for idx, postcard in enumerate(reversed(self.postcards)):
            html += f"""
            <div class="postcard-item" style="animation-delay: {idx * 0.1}s" onclick="openModal('{postcard['path']}')">
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
    
    <!-- Modal for full-size view -->
    <div id="myModal" class="modal" onclick="closeModal(event)">
        <span class="close" onclick="closeModal(event)">&times;</span>
        <img class="modal-content" id="modalImg">
        <a href="#" id="downloadLink" class="download-btn" download>⬇️ Download Postcard</a>
    </div>
    
    <script>
        function openModal(imagePath) {
            event.stopPropagation();
            const modal = document.getElementById('myModal');
            const modalImg = document.getElementById('modalImg');
            const downloadLink = document.getElementById('downloadLink');
            
            modal.style.display = 'flex';
            modalImg.src = imagePath;
            downloadLink.href = imagePath;
            
            // Prevent body scroll when modal is open
            document.body.style.overflow = 'hidden';
        }
        
        function closeModal(event) {
            event.stopPropagation();
            document.getElementById('myModal').style.display = 'none';
            document.body.style.overflow = 'auto';
        }
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal(event);
            }
        });
    </script>
</body>
</html>
"""
        
        # Save HTML file
        output_path = './postcards/my_postcards.html'
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"\n✅ Postcards page updated: {output_path}")
        print(f"   📮 Total postcards: {len(self.postcards)}")
        
        return output_path


# Demo usage
def main():
    """
    Demo: Create postcards from user photos
    """
    
    print("="*70)
    print("📮 POSTCARD GENERATOR - Using Your Photos!")
    print("="*70)
    
    # Initialize
    generator = PostcardGenerator()
    
    # Example 1: User at Durham
    print("\n" + "="*70)
    print("👤 USER ACTION: 'Click a picture!'")
    print("="*70)
    print("📸 *User takes photo of Durham building*")
    print("📍 Location detected: Durham, North Carolina")
    
    postcard1 = generator.create_postcard(
        photo_path="test_images/IMG_8540.JPG",  # USER'S ACTUAL PHOTO
        location_name="Durham, North Carolina"
    )
    
    if postcard1:
        print("✅ Postcard #1 created!")
    
    
    # Generate postcards gallery page
    print("\n" + "="*70)
    print("📱 Generating 'My Postcards' gallery page...")
    print("="*70)
    
    page_path = generator.generate_postcards_page()
    
    if page_path:
        print(f"\n🎉 SUCCESS!")
        print(f"📂 Open in browser: {page_path}")
        print(f"📮 Total postcards: {len(generator.postcards)}")
        print("\n💡 Each postcard:")
        print("   • Based on YOUR actual photo")
        print("   • Transformed into vintage 1940s travel poster style")
        print("   • Includes location name and date")
        print("   • Downloadable in high resolution")


if __name__ == "__main__":
    main()