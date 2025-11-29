#!/usr/bin/env python3
"""
Bank Card/Document Upload Test Demonstration
===========================================
Demonstrates how to test OCR system with real bank documents

Usage:
    python test_bank_card_upload.py [image_path]
    python test_bank_card_upload.py --demo  # Create and test sample images
"""

import sys
import time
import json
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from bank_ocr_api import OCRSystem


def create_realistic_bank_card():
    """Create a realistic bank card/cheque image for testing"""
    # Create card-sized image (standard cheque size)
    width, height = 850, 400
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to load a better font
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Bank header
    draw.rectangle([20, 20, 830, 80], outline='black', width=2)
    draw.text((50, 35), "NEPAL BANK LIMITED", fill='black', font=font_large)
    draw.text((600, 45), "Cheque No: 000123", fill='black', font=font_small)
    
    # Date field
    draw.text((650, 100), "Date:", fill='black', font=font_medium)
    draw.rectangle([700, 95, 800, 120], outline='black', width=1)
    draw.text((705, 100), "15/01/2024", fill='black', font=font_small)
    
    # Pay to line
    draw.text((50, 130), "Pay:", fill='black', font=font_medium)
    draw.text((100, 130), "Mr. Ram Bahadur Sharma", fill='black', font=font_medium)
    draw.line([100, 155, 600, 155], fill='black', width=1)
    
    # Amount in words
    draw.text((50, 180), "Rupees:", fill='black', font=font_medium)
    draw.text((130, 180), "Twenty Five Thousand Only", fill='black', font=font_medium)
    draw.line([130, 205, 700, 205], fill='black', width=1)
    
    # Amount box
    draw.rectangle([700, 165, 800, 210], outline='black', width=2)
    draw.text((710, 180), "Rs.25,000/-", fill='black', font=font_medium)
    
    # Account number
    draw.text((50, 240), "A/C No:", fill='black', font=font_medium)
    draw.text((120, 240), "00-1234-5678-9012", fill='black', font=font_medium)
    
    # Bank details
    draw.text((50, 270), "Branch: Kathmandu Main", fill='black', font=font_small)
    draw.text((50, 290), "SWIFT: NBLBNPKA", fill='black', font=font_small)
    
    # Signature line
    draw.text((550, 320), "Authorized Signature", fill='black', font=font_small)
    draw.line([520, 350, 750, 350], fill='black', width=1)
    
    # MICR line (bottom)
    draw.text((200, 370), "||:123456||: 789012345678||: 67||", fill='black', font=font_small)
    
    # Add some security features (watermark-like)
    draw.text((400, 200), "SPECIMEN", fill='lightgray', font=font_large)
    
    return img


def create_deposit_slip():
    """Create a deposit slip for testing"""
    img = Image.new('RGB', (600, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_text = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Header
    draw.text((200, 30), "BANK DEPOSIT SLIP", fill='black', font=font_title)
    draw.text((180, 60), "Nabil Bank Limited", fill='black', font=font_text)
    
    # Form fields
    y_pos = 120
    fields = [
        ("Branch:", "Kathmandu"),
        ("Date:", "15/01/2024"),
        ("A/C Number:", "9876-5432-1098-7654"),
        ("A/C Name:", "Ms. Sita Devi Sharma"),
        ("", ""),
        ("DEPOSIT BREAKDOWN:", ""),
        ("Cash:", "Rs. 10,000.00"),
        ("Cheque:", "Rs. 15,000.00"),
        ("Draft:", "Rs. 0.00"),
        ("", ""),
        ("TOTAL:", "Rs. 25,000.00"),
        ("", ""),
        ("Depositor:", "Ram Kumar"),
        ("Phone:", "9841-234567"),
    ]
    
    for label, value in fields:
        if label:
            draw.text((50, y_pos), label, fill='black', font=font_text)
            if value:
                draw.text((200, y_pos), value, fill='black', font=font_text)
        y_pos += 30
        
        if label == "TOTAL:":
            draw.line([200, y_pos - 10], [400, y_pos - 10], fill='black', width=2)
    
    # Signature boxes
    draw.rectangle([50, 650], [250, 700], outline='black', width=1)
    draw.text((60, 660), "Depositor's Signature", fill='black', font=font_small)
    
    draw.rectangle([350, 650], [550, 700], outline='black', width=1)
    draw.text((360, 660), "Bank Official", fill='black', font=font_small)
    
    # Border
    draw.rectangle([30, 30], [570, 750], outline='black', width=2)
    
    return img


def test_ocr_with_image(image_path, use_api=False):
    """Test OCR system with a specific image"""
    print(f"Testing OCR with image: {image_path}")
    print("-" * 50)
    
    if use_api:
        # Test via API
        return test_via_api(image_path)
    else:
        # Test directly with OCR system
        return test_direct_ocr(image_path)


def test_direct_ocr(image_path):
    """Test using OCR system directly"""
    try:
        # Load image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            # Assume it's already a numpy array
            image = image_path
        
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        
        # Initialize OCR system
        print("Initializing OCR system...")
        ocr_system = OCRSystem()
        ocr_system.load_models()
        
        # Process document
        start_time = time.time()
        result = ocr_system.process_document(image)
        processing_time = time.time() - start_time
        
        # Display results
        print(f"Processing time: {processing_time:.2f} seconds")
        print("\nExtracted Fields:")
        print("=" * 50)
        
        for field_name, field_data in result.items():
            value = field_data.get("value", "")
            confidence = field_data.get("confidence", 0.0)
            
            if value:
                print(f"{field_name:20}: {value} (confidence: {confidence:.3f})")
            else:
                print(f"{field_name:20}: [Not detected]")
        
        return True, result
        
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return False, str(e)


def test_via_api(image_path):
    """Test using the FastAPI endpoint"""
    try:
        # Check if API is running
        api_url = "http://localhost:8000"
        
        # Health check
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code != 200:
                print("API health check failed. Make sure the API server is running:")
                print("python bank_ocr_api.py")
                return False, "API not available"
        except requests.exceptions.ConnectionError:
            print("Could not connect to API. Make sure the server is running on port 8000:")
            print("python bank_ocr_api.py")
            return False, "API not available"
        
        # Upload file
        if isinstance(image_path, str) or isinstance(image_path, Path):
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/png')}
                response = requests.post(f"{api_url}/upload-document/", files=files)
        else:
            # Convert numpy array to bytes
            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            files = {'file': ('test_image.png', img_bytes, 'image/png')}
            response = requests.post(f"{api_url}/upload-document/", files=files)
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            print(f"API Response Status: SUCCESS")
            print(f"Processing Time: {result.get('processingTime', 'N/A')} seconds")
            
            print("\nExtracted Fields (via API):")
            print("=" * 50)
            
            extracted_data = result.get('data', {})
            for field_name, field_data in extracted_data.items():
                value = field_data.get("value", "")
                confidence = field_data.get("confidence", 0.0)
                
                if value:
                    print(f"{field_name:20}: {value} (confidence: {confidence:.3f})")
                else:
                    print(f"{field_name:20}: [Not detected]")
            
            return True, result
        else:
            error_data = response.json()
            print(f"API Error ({response.status_code}): {error_data.get('error', 'Unknown')}")
            print(f"Message: {error_data.get('message', '')}")
            return False, error_data
            
    except Exception as e:
        print(f"Error testing via API: {e}")
        return False, str(e)


def save_test_images():
    """Create and save test images"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    print("Creating test images...")
    
    # Create bank card/cheque
    cheque = create_realistic_bank_card()
    cheque_path = test_dir / "sample_bank_cheque.png"
    cheque.save(cheque_path)
    print(f"✓ Created: {cheque_path}")
    
    # Create deposit slip
    deposit = create_deposit_slip()
    deposit_path = test_dir / "sample_deposit_slip.png"
    deposit.save(deposit_path)
    print(f"✓ Created: {deposit_path}")
    
    return [cheque_path, deposit_path]


def main():
    parser = argparse.ArgumentParser(description="Test Bank OCR System")
    parser.add_argument('image_path', nargs='?', help='Path to bank document image')
    parser.add_argument('--demo', action='store_true', help='Create and test demo images')
    parser.add_argument('--api', action='store_true', help='Test via API (requires server running)')
    parser.add_argument('--create-only', action='store_true', help='Only create test images, don\'t process')
    
    args = parser.parse_args()
    
    if args.demo or not args.image_path:
        # Create demo images
        test_images = save_test_images()
        
        if args.create_only:
            print("\nTest images created successfully!")
            return
        
        print("\nTesting with created images...")
        
        for image_path in test_images:
            print(f"\n{'='*60}")
            print(f"Testing: {image_path.name}")
            print('='*60)
            
            success, result = test_ocr_with_image(image_path, use_api=args.api)
            
            if success:
                print("✓ Test completed successfully!")
            else:
                print(f"✗ Test failed: {result}")
            
            print()  # Add spacing
    
    elif args.image_path:
        # Test specific image
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return 1
        
        success, result = test_ocr_with_image(image_path, use_api=args.api)
        
        if success:
            print("\n✓ Test completed successfully!")
            
            # Save result to file
            result_file = image_path.parent / f"{image_path.stem}_ocr_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {result_file}")
        else:
            print(f"\n✗ Test failed: {result}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())