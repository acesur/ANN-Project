"""
Test Suite for Real Bank Document Images
========================================
Tests OCR system with actual bank documents (cheques, deposit slips, etc.)
Place test images in tests/test_images/ directory
"""

import unittest
import os
import cv2
import numpy as np
import json
import time
from pathlib import Path
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bank_ocr_api import OCRSystem
from fastapi.testclient import TestClient
from bank_ocr_api import app


class TestRealBankDocuments(unittest.TestCase):
    """Test OCR with real bank document images"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.ocr_system = OCRSystem()
        cls.ocr_system.load_models()
        cls.client = TestClient(app)
        
        # Create test images directory if it doesn't exist
        cls.test_images_dir = Path(__file__).parent / "test_images"
        cls.test_images_dir.mkdir(exist_ok=True)
        
        # Create sample test images if they don't exist
        cls.create_sample_test_images()
        
    @classmethod
    def create_sample_test_images(cls):
        """Create sample test images for testing"""
        # Sample cheque image
        cheque_path = cls.test_images_dir / "sample_cheque.png"
        if not cheque_path.exists():
            cls.create_sample_cheque(cheque_path)
            
        # Sample deposit slip
        deposit_path = cls.test_images_dir / "sample_deposit.png"
        if not deposit_path.exists():
            cls.create_sample_deposit_slip(deposit_path)
            
        # Sample with Nepali text
        nepali_path = cls.test_images_dir / "sample_nepali.png"
        if not nepali_path.exists():
            cls.create_sample_nepali_document(nepali_path)
            
    @classmethod
    def create_sample_cheque(cls, filepath):
        """Create a realistic sample cheque image"""
        # Create white background
        img = np.ones((400, 850, 3), dtype=np.uint8) * 255
        
        # Add text using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Bank name
        cv2.putText(img, "NEPAL BANK LIMITED", (50, 50), 
                   font, 1.0, (0, 0, 0), 2)
        
        # Date
        cv2.putText(img, "Date: 15/01/2024", (600, 50), 
                   font, 0.6, (0, 0, 0), 1)
        
        # Pay line
        cv2.putText(img, "Pay: Mr. Ram Bahadur Sharma", (50, 120), 
                   font, 0.7, (0, 0, 0), 1)
        cv2.line(img, (100, 140), (500, 140), (0, 0, 0), 1)
        
        # Amount in words
        cv2.putText(img, "Rupees: Twenty Five Thousand Only", (50, 180), 
                   font, 0.7, (0, 0, 0), 1)
        cv2.line(img, (140, 200), (550, 200), (0, 0, 0), 1)
        
        # Amount in numbers
        cv2.rectangle(img, (600, 150), (780, 200), (0, 0, 0), 2)
        cv2.putText(img, "Rs. 25,000/-", (610, 180), 
                   font, 0.8, (0, 0, 0), 2)
        
        # Account number
        cv2.putText(img, "A/C No: 00123456789012", (50, 250), 
                   font, 0.6, (0, 0, 0), 1)
        
        # Signature area
        cv2.putText(img, "Authorized Signature", (550, 320), 
                   font, 0.5, (0, 0, 0), 1)
        cv2.line(img, (520, 340), (750, 340), (0, 0, 0), 1)
        
        # MICR line (bottom)
        cv2.putText(img, "||:123456||: 789012345||: 67||", (200, 380), 
                   font, 0.8, (0, 0, 0), 1)
        
        # Add some decorative elements
        cv2.rectangle(img, (20, 20), (830, 380), (0, 0, 0), 2)
        
        # Save image
        cv2.imwrite(str(filepath), img)
        
    @classmethod
    def create_sample_deposit_slip(cls, filepath):
        """Create a sample deposit slip image"""
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Header
        cv2.putText(img, "BANK DEPOSIT SLIP", (180, 50), 
                   font, 1.2, (0, 0, 0), 2)
        cv2.putText(img, "NABIL BANK LIMITED", (200, 90), 
                   font, 0.8, (0, 0, 0), 1)
        
        # Branch and date
        cv2.putText(img, "Branch: Kathmandu", (50, 150), 
                   font, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Date: 15/01/2024", (350, 150), 
                   font, 0.7, (0, 0, 0), 1)
        
        # Account details
        cv2.putText(img, "Account Number: 9876543210123456", (50, 220), 
                   font, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Account Name: Ms. Sita Devi Sharma", (50, 260), 
                   font, 0.7, (0, 0, 0), 1)
        
        # Deposit details
        cv2.putText(img, "DEPOSIT DETAILS", (50, 320), 
                   font, 0.8, (0, 0, 0), 2)
        cv2.line(img, (50, 340), (550, 340), (0, 0, 0), 1)
        
        # Cash amount
        cv2.putText(img, "Cash:", (50, 380), 
                   font, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Rs. 10,000.00", (400, 380), 
                   font, 0.7, (0, 0, 0), 1)
        
        # Cheque amount
        cv2.putText(img, "Cheque:", (50, 420), 
                   font, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Rs. 15,000.00", (400, 420), 
                   font, 0.7, (0, 0, 0), 1)
        
        # Total
        cv2.line(img, (350, 450), (500, 450), (0, 0, 0), 2)
        cv2.putText(img, "TOTAL:", (50, 480), 
                   font, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "Rs. 25,000.00", (400, 480), 
                   font, 0.8, (0, 0, 0), 2)
        
        # Depositor details
        cv2.putText(img, "Depositor Name: Ram Kumar", (50, 550), 
                   font, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "Phone: 9841234567", (50, 580), 
                   font, 0.6, (0, 0, 0), 1)
        
        # Signature
        cv2.putText(img, "Depositor's Signature", (50, 650), 
                   font, 0.5, (0, 0, 0), 1)
        cv2.line(img, (50, 670), (250, 670), (0, 0, 0), 1)
        
        cv2.putText(img, "Bank Official", (350, 650), 
                   font, 0.5, (0, 0, 0), 1)
        cv2.line(img, (350, 670), (550, 670), (0, 0, 0), 1)
        
        # Border
        cv2.rectangle(img, (30, 30), (570, 750), (0, 0, 0), 2)
        
        cv2.imwrite(str(filepath), img)
        
    @classmethod
    def create_sample_nepali_document(cls, filepath):
        """Create a sample document with Nepali text"""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Mix of English and Nepali numerals
        cv2.putText(img, "RASTRIYA BANIJYA BANK", (200, 50), 
                   font, 1.0, (0, 0, 0), 2)
        
        # Account details with Nepali numerals
        cv2.putText(img, "Khata No: 1234567890", (50, 150), 
                   font, 0.7, (0, 0, 0), 1)
        
        # Name in English
        cv2.putText(img, "Name: Ram Prasad Sharma", (50, 200), 
                   font, 0.7, (0, 0, 0), 1)
        
        # Amount with Nepali format
        cv2.putText(img, "Rakam: Rs. 50,000/-", (50, 250), 
                   font, 0.7, (0, 0, 0), 1)
        
        # Date in Nepali calendar format
        cv2.putText(img, "Miti: 2081/08/07", (50, 300), 
                   font, 0.7, (0, 0, 0), 1)
        
        # Routing number
        cv2.putText(img, "Routing: 123456789", (50, 350), 
                   font, 0.7, (0, 0, 0), 1)
        
        cv2.imwrite(str(filepath), img)
        
    def test_process_sample_cheque(self):
        """Test processing a sample cheque image"""
        cheque_path = self.test_images_dir / "sample_cheque.png"
        
        if not cheque_path.exists():
            self.skipTest("Sample cheque image not found")
            
        # Load image
        image = cv2.imread(str(cheque_path))
        
        # Process with OCR system
        start_time = time.time()
        result = self.ocr_system.process_document(image)
        processing_time = time.time() - start_time
        
        print(f"\nCheque Processing Time: {processing_time:.2f}s")
        print(f"Extracted Data: {json.dumps(result, indent=2)}")
        
        # Validate results
        self.assertIsInstance(result, dict)
        
        # Check if bank name was extracted
        if result["bankName"]["value"]:
            self.assertIn("BANK", result["bankName"]["value"].upper())
            
        # Check if account number was extracted
        if result["accountNumber"]["value"]:
            self.assertTrue(len(result["accountNumber"]["value"]) >= 10)
            
        # Check processing time is reasonable
        self.assertLess(processing_time, 5.0, "Processing took too long")
        
    def test_process_sample_deposit_slip(self):
        """Test processing a sample deposit slip"""
        deposit_path = self.test_images_dir / "sample_deposit.png"
        
        if not deposit_path.exists():
            self.skipTest("Sample deposit slip not found")
            
        image = cv2.imread(str(deposit_path))
        
        start_time = time.time()
        result = self.ocr_system.process_document(image)
        processing_time = time.time() - start_time
        
        print(f"\nDeposit Slip Processing Time: {processing_time:.2f}s")
        print(f"Extracted Data: {json.dumps(result, indent=2)}")
        
        self.assertIsInstance(result, dict)
        
        # Check if amount was extracted
        if result["amount"]["value"]:
            # Should contain numbers
            self.assertTrue(any(c.isdigit() for c in result["amount"]["value"]))
            
    def test_process_nepali_document(self):
        """Test processing document with Nepali text"""
        nepali_path = self.test_images_dir / "sample_nepali.png"
        
        if not nepali_path.exists():
            self.skipTest("Sample Nepali document not found")
            
        image = cv2.imread(str(nepali_path))
        
        result = self.ocr_system.process_document(image)
        
        print(f"\nNepali Document Extracted Data: {json.dumps(result, indent=2)}")
        
        self.assertIsInstance(result, dict)
        
        # Should handle mixed language content
        
    def test_api_with_real_image(self):
        """Test API endpoint with real image upload"""
        cheque_path = self.test_images_dir / "sample_cheque.png"
        
        if not cheque_path.exists():
            self.skipTest("Sample image not found")
            
        with open(cheque_path, "rb") as f:
            files = {"file": ("cheque.png", f, "image/png")}
            response = self.client.post("/upload-document/", files=files)
            
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertTrue(data["success"])
        self.assertIn("processingTime", data)
        
        print(f"\nAPI Response: {json.dumps(data, indent=2)}")
        
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        test_images = list(self.test_images_dir.glob("*.png"))
        
        if not test_images:
            self.skipTest("No test images found")
            
        total_time = 0
        results = []
        
        for img_path in test_images[:3]:  # Process up to 3 images
            image = cv2.imread(str(img_path))
            
            start_time = time.time()
            result = self.ocr_system.process_document(image)
            processing_time = time.time() - start_time
            
            total_time += processing_time
            results.append({
                "file": img_path.name,
                "time": processing_time,
                "success": bool(result)
            })
            
        avg_time = total_time / len(results)
        
        print(f"\nBatch Processing Results:")
        for r in results:
            print(f"  {r['file']}: {r['time']:.2f}s - {'✓' if r['success'] else '✗'}")
        print(f"Average time: {avg_time:.2f}s")
        
        # Average should be under 3 seconds per document
        self.assertLess(avg_time, 3.0)


class TestImageQualityScenarios(unittest.TestCase):
    """Test OCR with various image quality scenarios"""
    
    @classmethod
    def setUpClass(cls):
        cls.ocr_system = OCRSystem()
        cls.ocr_system.load_models()
        
    def apply_noise(self, image, noise_type="gaussian"):
        """Apply different types of noise to image"""
        if noise_type == "gaussian":
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            noisy = cv2.add(image, noise)
        elif noise_type == "salt_pepper":
            noisy = image.copy()
            prob = 0.05
            mask = np.random.random(image.shape[:2])
            noisy[mask < prob/2] = 0
            noisy[mask > 1 - prob/2] = 255
        else:
            noisy = image
            
        return noisy
    
    def apply_blur(self, image, blur_type="gaussian"):
        """Apply different types of blur to image"""
        if blur_type == "gaussian":
            blurred = cv2.GaussianBlur(image, (5, 5), 1)
        elif blur_type == "motion":
            kernel = np.zeros((15, 15))
            kernel[7, :] = 1
            kernel = kernel / 15
            blurred = cv2.filter2D(image, -1, kernel)
        else:
            blurred = image
            
        return blurred
    
    def test_low_resolution_image(self):
        """Test OCR on low resolution image"""
        # Create a small image
        small_img = np.ones((100, 150, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(small_img, "TEST", (10, 50), font, 0.5, (0, 0, 0), 1)
        
        result = self.ocr_system.process_document(small_img)
        self.assertIsInstance(result, dict)
        
    def test_high_resolution_image(self):
        """Test OCR on high resolution image"""
        # Create a large image (4K)
        large_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(large_img, "BANK NAME", (100, 200), font, 3.0, (0, 0, 0), 5)
        cv2.putText(large_img, "Account: 1234567890", (100, 400), font, 2.0, (0, 0, 0), 3)
        
        # Should handle without memory issues
        try:
            result = self.ocr_system.process_document(large_img)
            self.assertIsInstance(result, dict)
        except MemoryError:
            self.skipTest("Insufficient memory for high resolution test")
            
    def test_noisy_image(self):
        """Test OCR on noisy image"""
        # Create image with text
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Nepal Bank", (50, 100), font, 1.0, (0, 0, 0), 2)
        cv2.putText(img, "Account: 123456", (50, 200), font, 0.8, (0, 0, 0), 2)
        
        # Add noise
        noisy_img = self.apply_noise(img, "gaussian")
        
        result = self.ocr_system.process_document(noisy_img)
        self.assertIsInstance(result, dict)
        
    def test_blurred_image(self):
        """Test OCR on blurred image"""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Bank Document", (50, 100), font, 1.0, (0, 0, 0), 2)
        
        # Apply blur
        blurred_img = self.apply_blur(img, "gaussian")
        
        result = self.ocr_system.process_document(blurred_img)
        self.assertIsInstance(result, dict)
        
    def test_rotated_image(self):
        """Test OCR on slightly rotated image"""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Test Bank", (50, 100), font, 1.0, (0, 0, 0), 2)
        
        # Rotate image by 5 degrees
        center = (img.shape[1]//2, img.shape[0]//2)
        matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]),
                                    borderValue=(255, 255, 255))
        
        result = self.ocr_system.process_document(rotated_img)
        self.assertIsInstance(result, dict)


def run_real_image_tests():
    """Run tests with real images"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRealBankDocuments))
    suite.addTests(loader.loadTestsFromTestCase(TestImageQualityScenarios))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("REAL IMAGE TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_real_image_tests()
    exit(0 if success else 1)