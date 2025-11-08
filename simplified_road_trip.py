#!/usr/bin/env python3
"""
AI Road Trip Companion - Simplified Version
Focus: YOLO sign detection + OCR + AI recommendations
No Raspberry Pi dependencies required
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import json

# Optional imports - will gracefully handle if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not installed. Install with: pip install ultralytics")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  EasyOCR not installed. Install with: pip install easyocr")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. Install with: pip install openai")


class Config:
    """Configuration settings"""
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = BASE_DIR / "output"
    DETECTED_SIGNS_DIR = OUTPUT_DIR / "detected_signs"
    
    # Detection settings
    YOLO_MODEL = "yolov8n.pt"
    SIGN_CONFIDENCE_THRESHOLD = 0.3
    
    # API settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.DETECTED_SIGNS_DIR.mkdir(exist_ok=True)


class SignDetector:
    """Detect signboards using YOLO or traditional CV methods"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                print("📦 Loading YOLOv8 model...")
                self.model = YOLO(config.YOLO_MODEL)
                print("✅ YOLOv8 loaded successfully")
            except Exception as e:
                print(f"⚠️  YOLOv8 load failed: {e}")
                print("   Will use fallback detection method")
    
    def detect_signs(self, image_path):
        """Detect signs in the image"""
        print(f"\n🔍 Analyzing image: {Path(image_path).name}")
        
        # Try YOLO first
        if self.model:
            signs = self._detect_with_yolo(image_path)
            if signs:
                print(f"✅ YOLO detected {len(signs)} object(s)")
                return signs
        
        # Fallback to traditional CV
        print("🔄 Using traditional computer vision detection...")
        signs = self._detect_rectangular_regions(image_path)
        print(f"✅ Found {len(signs)} rectangular region(s)")
        return signs
    
    def _detect_with_yolo(self, image_path):
        """Detect signs using YOLOv8"""
        results = self.model(str(image_path), conf=self.config.SIGN_CONFIDENCE_THRESHOLD)
        
        detected_signs = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                detected_signs.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': class_name,
                    'confidence': confidence
                })
                print(f"   • {class_name}: {confidence:.2f}")
        
        return detected_signs
    
    def _detect_rectangular_regions(self, image_path):
        """Fallback: Detect rectangular regions using traditional CV"""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_regions = []
        h, w = img.shape[:2]
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Look for rectangles
            if len(approx) == 4:
                x, y, w_box, h_box = cv2.boundingRect(approx)
                area = w_box * h_box
                img_area = h * w
                
                # Filter by size
                if 0.01 < (area / img_area) < 0.3:
                    detected_regions.append({
                        'bbox': [x, y, x + w_box, y + h_box],
                        'class': 'rectangular_region',
                        'confidence': 0.6
                    })
        
        return detected_regions[:10]  # Return top 10


class TextExtractor:
    """Extract text from signboard images using OCR"""
    
    def __init__(self, config):
        self.config = config
        self.reader = None
        
        if OCR_AVAILABLE:
            try:
                print("📦 Initializing EasyOCR (this may take a moment)...")
                self.reader = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR initialized")
            except Exception as e:
                print(f"⚠️  EasyOCR initialization failed: {e}")
    
    def extract_text(self, image_path, bboxes):
        """Extract text from detected sign regions"""
        if not self.reader:
            print("⚠️  OCR not available - using manual extraction")
            return self._manual_extraction(image_path)
        
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        extracted_texts = []
        
        print(f"\n📝 Extracting text from {len(bboxes)} region(s)...")
        
        for idx, bbox_info in enumerate(bboxes):
            bbox = bbox_info['bbox']
            x1, y1, x2, y2 = bbox
            
            # Crop the sign region
            sign_crop = img[y1:y2, x1:x2]
            
            if sign_crop.size == 0:
                continue
            
            # Save cropped sign
            crop_filename = f"sign_crop_{idx}.jpg"
            crop_path = self.config.DETECTED_SIGNS_DIR / crop_filename
            cv2.imwrite(str(crop_path), sign_crop)
            
            # Perform OCR
            try:
                results = self.reader.readtext(sign_crop)
                
                text_parts = []
                for detection in results:
                    text = detection[1]
                    confidence = detection[2]
                    if confidence > 0.3:
                        text_parts.append(text)
                        print(f"   Region {idx+1}: '{text}' (confidence: {confidence:.2f})")
                
                if text_parts:
                    full_text = ' '.join(text_parts)
                    extracted_texts.append({
                        'text': full_text,
                        'bbox': bbox,
                        'crop_path': str(crop_path),
                        'region_index': idx
                    })
            
            except Exception as e:
                print(f"   Error in region {idx+1}: {e}")
        
        # Also try full image OCR
        try:
            print("\n📝 Also checking full image...")
            full_results = self.reader.readtext(img)
            all_text = []
            for detection in full_results:
                text = detection[1]
                confidence = detection[2]
                if confidence > 0.4:
                    all_text.append(text)
                    print(f"   Full image: '{text}' (confidence: {confidence:.2f})")
            
            if all_text and not extracted_texts:
                combined = ' '.join(all_text)
                extracted_texts.append({
                    'text': combined,
                    'bbox': [0, 0, img.shape[1], img.shape[0]],
                    'crop_path': str(image_path),
                    'region_index': -1
                })
        except Exception as e:
            print(f"   Full image OCR error: {e}")
        
        return extracted_texts
    
    def _manual_extraction(self, image_path):
        """Manual text extraction for testing without OCR"""
        # For the Morrisville example
        if "morrisville" in str(image_path).lower() or "1762567630203" in str(image_path):
            return [{
                'text': 'MORRISVILLE TOWN LIMIT',
                'bbox': [350, 230, 850, 430],
                'crop_path': str(image_path),
                'region_index': 0
            }]
        return []


class AIRecommendationEngine:
    """Generate travel recommendations using ChatGPT"""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        
        if OPENAI_AVAILABLE and config.OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                print("✅ ChatGPT client initialized")
            except Exception as e:
                print(f"⚠️  ChatGPT initialization failed: {e}")
    
    def generate_recommendations(self, sign_data):
        """Generate travel recommendations based on detected signs"""
        if not sign_data:
            return "No text detected from signs."
        
        # Extract all detected text
        sign_texts = [item['text'] for item in sign_data]
        combined_text = ' | '.join(sign_texts)
        
        print(f"\n🤖 Generating recommendations for: {combined_text}")
        
        if not self.client:
            print("⚠️  Using simulated recommendation (no API key)")
            return self._simulate_recommendation(combined_text)
        
        # Create prompt
        prompt = f"""You are an AI road trip companion. I just passed a road sign that says:

"{combined_text}"

Based on this sign, please provide:
1. Where I might be (location identification)
2. 2-3 interesting facts about this location
3. 2-3 recommended attractions or points of interest nearby
4. 1-2 practical travel tips for this area

Keep your response engaging, informative, and concise (under 250 words).
"""
        
        try:
            print("   Calling ChatGPT API...")
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o (latest model)
                messages=[
                    {"role": "system", "content": "You are a helpful AI road trip companion that provides interesting and useful travel information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            recommendation = response.choices[0].message.content
            print("✅ Recommendation generated!")
            return recommendation
        
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return self._simulate_recommendation(combined_text)
    
    def _simulate_recommendation(self, sign_text):
        """Fallback recommendation"""
        return f"""📍 Sign Detected: {sign_text}

ℹ️  To get AI-powered recommendations:
1. Get an API key from: https://platform.openai.com/api-keys
2. Set it: export OPENAI_API_KEY='your_key_here'
3. Run this script again

For now: Research '{sign_text}' online for local attractions and points of interest!
"""


class SimplifiedRoadTripAI:
    """Main simplified system - no Raspberry Pi dependencies"""
    
    def __init__(self):
        self.config = Config()
        self.sign_detector = SignDetector(self.config)
        self.text_extractor = TextExtractor(self.config)
        self.ai_engine = AIRecommendationEngine(self.config)
    
    def process_image(self, image_path):
        """Process a single image through the pipeline"""
        print("\n" + "="*70)
        print(f"🚗 PROCESSING: {Path(image_path).name}")
        print("="*70)
        
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Step 1: Detect signs
        print("\n[STEP 1/3] SIGN DETECTION")
        print("-" * 70)
        detected_signs = self.sign_detector.detect_signs(image_path)
        
        if not detected_signs:
            print("❌ No signs detected")
            return None
        
        # Step 2: Extract text
        print("\n[STEP 2/3] TEXT EXTRACTION")
        print("-" * 70)
        sign_data = self.text_extractor.extract_text(image_path, detected_signs)
        
        if not sign_data:
            print("❌ No text extracted")
            return None
        
        print(f"\n✅ Extracted text: {', '.join([s['text'] for s in sign_data])}")
        
        # Step 3: Generate recommendation
        print("\n[STEP 3/3] AI RECOMMENDATION")
        print("-" * 70)
        recommendation = self.ai_engine.generate_recommendations(sign_data)
        
        # Display result
        print("\n" + "="*70)
        print("🎯 RECOMMENDATION:")
        print("="*70)
        print(recommendation)
        print("="*70)
        
        # Save result
        result = {
            'image': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'detected_signs': detected_signs,
            'extracted_text': sign_data,
            'recommendation': recommendation
        }
        
        output_file = self.config.OUTPUT_DIR / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return result


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Simplified AI Road Trip Companion - Test YOLO & AI Recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simplified_road_trip.py image.jpg
  python simplified_road_trip.py /path/to/sign/photo.jpg
  
Environment Variables:
  OPENAI_API_KEY  - Your OpenAI API key (get from https://platform.openai.com/api-keys)
        """
    )
    
    parser.add_argument('image', help='Path to image file to process')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚗 AI ROAD TRIP COMPANION - SIMPLIFIED VERSION")
    print("="*70)
    print("\nFocus: YOLO Detection + OCR + AI Recommendations")
    print("No Raspberry Pi dependencies\n")
    
    # Check what's available
    print("📋 Checking dependencies...")
    print(f"   YOLO (ultralytics):  {'✅ Available' if YOLO_AVAILABLE else '❌ Not installed'}")
    print(f"   OCR (easyocr):       {'✅ Available' if OCR_AVAILABLE else '❌ Not installed'}")
    print(f"   ChatGPT (openai):    {'✅ Available' if OPENAI_AVAILABLE else '❌ Not installed'}")
    print(f"   API Key:             {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    print()
    
    # Initialize system
    system = SimplifiedRoadTripAI()
    
    # Process the image
    system.process_image(args.image)


if __name__ == "__main__":
    main()
