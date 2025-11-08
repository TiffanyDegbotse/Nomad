#!/usr/bin/env python3
"""
AI Journey Reconstruction - Pure OpenAI Vision Pipeline
No YOLO, no EasyOCR - just OpenAI Vision API
"""

import os
import base64
import json
from datetime import datetime
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PIL import Image as PILImage
from PIL.ExifTags import TAGS
import openai
import re


class OpenAIImageProcessor:
    """Process images using OpenAI Vision API only"""
    
    def __init__(self, api_key=None):
        self.api_key = ""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required! Set OPENAI_API_KEY environment variable"
            )
        
        self.client = openai.OpenAI(api_key=self.api_key)
        print("✅ OpenAI Vision client initialized")
    
    def encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def detect_and_extract_text(self, image_path):
        """
        Use OpenAI Vision to detect signs and extract text
        Single API call for everything
        """
        print(f"   🔍 Analyzing with OpenAI Vision...")
        
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Prompt for sign detection + OCR
        prompt = """Analyze this image and identify any road signs, street signs, town signs, or signboards that indicate location information.

For each sign you detect, provide:
1. The exact text on the sign
2. The type of sign (welcome sign, town limit, city limit, street sign, highway sign, exit sign, etc.)
3. Your confidence level (0.0 to 1.0)

Respond ONLY with valid JSON in this exact format:
{
  "signs": [
    {
      "text": "exact text on sign",
      "sign_type": "type of sign",
      "confidence": 0.95
    }
  ]
}

If no location signs are visible, return: {"signs": []}

IMPORTANT: 
- Only detect signs that have location information (city names, town names, street names, highway numbers)
- Ignore signs like "STOP", "YIELD", speed limit signs unless they have location info
- Extract text exactly as written on the sign
- Be confident only if you can clearly read the text
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"  # High detail for better OCR
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1  # Low temperature for accuracy
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON (handle markdown code blocks)
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            result = json.loads(result_text)
            
            detections = []
            if 'signs' in result and result['signs']:
                for sign in result['signs']:
                    if sign.get('text'):
                        detections.append({
                            'text': sign['text'],
                            'confidence': sign.get('confidence', 0.8),
                            'sign_type': sign.get('sign_type', 'unknown'),
                            'method': 'openai_vision'
                        })
                        print(f"      📝 '{sign['text']}' "
                              f"({sign.get('sign_type')}, "
                              f"conf: {sign.get('confidence', 0.8):.2f})")
            
            if not detections:
                print(f"      ℹ️  No location signs detected")
            
            return {
                'image_path': str(image_path),
                'has_signs': len(detections) > 0,
                'detections': detections
            }
            
        except json.JSONDecodeError as e:
            print(f"      ⚠️  Failed to parse response as JSON")
            print(f"      Response was: {result_text[:200]}")
            return {
                'image_path': str(image_path),
                'has_signs': False,
                'detections': [],
                'error': 'json_parse_error'
            }
        except Exception as e:
            print(f"      ❌ OpenAI API error: {e}")
            return {
                'image_path': str(image_path),
                'has_signs': False,
                'detections': [],
                'error': str(e)
            }


class JourneyDatabase:
    """Manage journey data in vector database"""
    
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
    
    def add_location(self, location_text, timestamp, image_id, 
                     confidence, has_sign=True, additional_metadata=None):
        """Add a detected location to the vector database"""
        
        # Generate embedding
        embedding = self.embedding_model.encode(location_text).tolist()
        
        # Prepare metadata
        metadata = {
            'timestamp': timestamp,
            'image_id': str(image_id),
            'location_text': location_text,
            'confidence': float(confidence),
            'has_sign': has_sign,
            'detection_type': 'sign_detected' if has_sign else 'inferred'
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Generate unique ID
        clean_timestamp = timestamp.replace(':', '').replace('-', '').replace('T', '_')
        doc_id = f"loc_{image_id}_{clean_timestamp}"
        
        try:
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[location_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            print(f"      ✅ Added to vector DB")
            return doc_id
        except Exception as e:
            print(f"      ⚠️  DB error: {e}")
            return None


def get_image_timestamp(image_path):
    """
    Extract timestamp from image
    Priority: 1) EXIF data, 2) Filename pattern, 3) File modification time
    """
    
    # Try EXIF data
    try:
        img = PILImage.open(image_path)
        exif_data = img.getexif()
        
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTime' or tag == 'DateTimeOriginal':
                    # Format: '2025:01:07 10:30:45'
                    dt_str = str(value).replace(':', '-', 2)  # Fix date part
                    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                    return dt.isoformat()
    except Exception:
        pass
    
    # Try filename pattern (e.g., IMG_20250107_103045.jpg)
    try:
        filename = Path(image_path).stem
        
        # Pattern: YYYYMMDD_HHMMSS
        match = re.search(r'(\d{8})_(\d{6})', filename)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            dt = datetime.strptime(f"{date_str}{time_str}", '%Y%m%d%H%M%S')
            return dt.isoformat()
        
        # Pattern: YYYY-MM-DD_HH-MM-SS
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', filename)
        if match:
            dt = datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), int(match.group(5)), int(match.group(6))
            )
            return dt.isoformat()
    except Exception:
        pass
    
    # Fallback: File modification time
    try:
        mod_time = os.path.getmtime(image_path)
        dt = datetime.fromtimestamp(mod_time)
        return dt.isoformat()
    except Exception:
        # Last resort: current time with incrementing seconds
        return datetime.now().isoformat()


def process_images_folder(folder_path, processor, database):
    """
    Process all images in a folder
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in folder.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    # Sort by filename (usually chronological)
    image_files.sort()
    
    print(f"\n📂 Found {len(image_files)} images in {folder_path}")
    print("="*70)
    
    if not image_files:
        print("⚠️  No images found in folder!")
        return []
    
    results = []
    
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] {image_path.name}")
        
        # Get timestamp
        timestamp = get_image_timestamp(image_path)
        print(f"   ⏰ Timestamp: {timestamp}")
        
        # Process image with OpenAI
        result = processor.detect_and_extract_text(image_path)
        
        # Add to database if signs detected
        if result['has_signs']:
            for detection in result['detections']:
                database.add_location(
                    location_text=detection['text'],
                    timestamp=timestamp,
                    image_id=idx,
                    confidence=detection['confidence'],
                    has_sign=True,
                    additional_metadata={
                        'image_name': image_path.name,
                        'method': 'openai_vision',
                        'sign_type': detection.get('sign_type', 'unknown')
                    }
                )
        
        results.append({
            'image_id': idx,
            'image_name': image_path.name,
            'timestamp': timestamp,
            'has_signs': result['has_signs'],
            'num_detections': len(result['detections']),
            'detections': result['detections']
        })
    
    return results


def display_summary(results):
    """Display processing summary"""
    
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    
    total_images = len(results)
    images_with_signs = sum(1 for r in results if r['has_signs'])
    total_detections = sum(r['num_detections'] for r in results)
    
    print(f"\n📊 Statistics:")
    print(f"   Total images processed: {total_images}")
    print(f"   Images with signs: {images_with_signs} ({images_with_signs/total_images*100:.1f}%)")
    print(f"   Total sign detections: {total_detections}")
    
    # Show detected locations chronologically
    print(f"\n📍 Detected Locations (chronological):")
    for r in results:
        if r['has_signs']:
            for det in r['detections']:
                print(f"   • {r['timestamp'][:19]} - {det['text']} "
                      f"(conf: {det['confidence']:.2f})")
    
    # Show unique locations
    print(f"\n🗺️  Unique Locations:")
    unique_locations = set()
    for r in results:
        if r['has_signs']:
            for det in r['detections']:
                unique_locations.add(det['text'])
    
    for loc in sorted(unique_locations):
        print(f"   • {loc}")
    
    # Estimate API cost
    print(f"\n💰 Estimated API Cost:")
    print(f"   Images processed: {total_images}")
    print(f"   Estimated cost: ${total_images * 0.01:.2f}")


def main():
    """Main execution"""
    
    print("="*70)
    print("🚗 AI JOURNEY RECONSTRUCTION")
    print("Pure OpenAI Vision Pipeline - No YOLO, No EasyOCR")
    print("="*70)
    
    # Configuration
    IMAGES_FOLDER = "./test_images"  # Change this to your images folder
    
    # Check for images folder
    if not Path(IMAGES_FOLDER).exists():
        print(f"\n⚠️  Images folder not found: {IMAGES_FOLDER}")
        print(f"Creating folder...")
        Path(IMAGES_FOLDER).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created folder: {IMAGES_FOLDER}")
        print(f"\n📸 Please add your road trip images to this folder and run again.")
        return
    
    # Initialize vector database
    print("\n📦 Step 1: Initializing vector database...")
    client = chromadb.PersistentClient(path="./journey_db")
    
    # Get or create collection
    try:
        collection = client.get_collection(name="journey_locations")
        print("   ℹ️  Using existing collection")
    except:
        collection = client.create_collection(name="journey_locations")
        print("   ℹ️  Created new collection")
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    database = JourneyDatabase(collection, embedding_model)
    print("   ✅ Database ready")
    
    # Initialize OpenAI processor
    print("\n📦 Step 2: Initializing OpenAI Vision processor...")
    try:
        processor = OpenAIImageProcessor()
    except ValueError as e:
        print(f"   ❌ {e}")
        print("\n   To fix this:")
        print("   1. Get API key from: https://platform.openai.com/api-keys")
        print("   2. Run: export OPENAI_API_KEY='your-key-here'")
        print("   3. Or set in code: processor = OpenAIImageProcessor(api_key='your-key')")
        return
    
    # Process images
    print(f"\n📸 Step 3: Processing images from: {IMAGES_FOLDER}")
    results = process_images_folder(IMAGES_FOLDER, processor, database)
    
    if not results:
        print("\n⚠️  No images were processed. Add images to the folder and try again.")
        return
    
    # Display summary
    display_summary(results)
    
    # Save results to JSON
    output_file = 'processing_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")
    
    print("\n✅ Pipeline complete!")
    print("\nNext steps:")
    print("   • Use the vector database for journey reconstruction")
    print("   • Query locations by similarity or time")
    print("   • Generate journey maps and narratives")


if __name__ == "__main__":
    main()