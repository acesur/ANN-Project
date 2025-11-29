"""
Integration Tests for Bank OCR API
===================================
Tests API endpoints with actual image uploads and end-to-end workflows
"""

import unittest
import asyncio
import json
import io
import base64
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app
from bank_ocr_api import app


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for Bank OCR API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        cls.client = TestClient(app)
        
    def create_test_image(self, text_content=None, format="PNG"):
        """Create a test image with optional text content"""
        # Create a white image
        img = Image.new('RGB', (800, 600), color='white')
        
        if text_content:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Try to use a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            # Draw text content
            y_position = 50
            for line in text_content:
                draw.text((50, y_position), line, fill='black', font=font)
                y_position += 50
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)
        
        return img_bytes
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("status", data)
        self.assertIn("message", data)
        self.assertIn("models_loaded", data)
        self.assertIn("version", data)
        
    def test_root_endpoint(self):
        """Test root endpoint for API information"""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("endpoints", data)
        self.assertIn("supported_formats", data)
        self.assertIn("supported_fields", data)
        
    def test_upload_valid_image(self):
        """Test uploading a valid image"""
        # Create test image with bank-like content
        test_content = [
            "Nepal Bank Limited",
            "Account: 1234567890123456",
            "Name: Ram Prasad",
            "Amount: Rs.50000",
            "Date: 2024-01-15"
        ]
        
        img_bytes = self.create_test_image(test_content, "PNG")
        
        files = {"file": ("test_document.png", img_bytes, "image/png")}
        response = self.client.post("/upload-document/", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check response structure
        self.assertTrue(data["success"])
        self.assertIn("data", data)
        self.assertIn("processingTime", data)
        
        # Check extracted fields structure
        extracted_data = data["data"]
        required_fields = ["bankName", "accountHolderName", "accountNumber", 
                          "routingNumber", "amount", "date"]
        
        for field in required_fields:
            self.assertIn(field, extracted_data)
            self.assertIn("value", extracted_data[field])
            self.assertIn("confidence", extracted_data[field])
            
    def test_upload_invalid_format(self):
        """Test uploading file with invalid format"""
        # Create a text file instead of image
        files = {"file": ("test.txt", b"This is a text file", "text/plain")}
        response = self.client.post("/upload-document/", files=files)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "INVALID_FILE_FORMAT")
        
    def test_upload_large_file(self):
        """Test uploading file exceeding size limit"""
        # Create a large file (>10MB)
        large_data = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large.jpg", large_data, "image/jpeg")}
        
        response = self.client.post("/upload-document/", files=files)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "FILE_TOO_LARGE")
        
    def test_upload_corrupted_image(self):
        """Test uploading corrupted image data"""
        # Send random bytes as JPEG
        corrupted_data = b"Not really a JPEG file content"
        files = {"file": ("corrupted.jpg", corrupted_data, "image/jpeg")}
        
        response = self.client.post("/upload-document/", files=files)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "CORRUPTED_IMAGE")
        
    def test_upload_empty_file(self):
        """Test uploading empty file"""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        
        response = self.client.post("/upload-document/", files=files)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        
        self.assertFalse(data["success"])
        # Should handle empty file appropriately
        
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        
        def make_request():
            img_bytes = self.create_test_image(["Test Bank", "Account: 123456"])
            files = {"file": ("test.png", img_bytes, "image/png")}
            return self.client.post("/upload-document/", files=files)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should complete successfully
        for response in responses:
            self.assertIn(response.status_code, [200, 400])  # Success or expected errors
            
    def test_processing_time_measurement(self):
        """Test that processing time is measured and returned"""
        img_bytes = self.create_test_image(["Test Content"])
        files = {"file": ("test.png", img_bytes, "image/png")}
        
        response = self.client.post("/upload-document/", files=files)
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("processingTime", data)
            self.assertIsInstance(data["processingTime"], (int, float))
            self.assertGreater(data["processingTime"], 0)
            
    def test_confidence_scores_range(self):
        """Test that confidence scores are in valid range"""
        img_bytes = self.create_test_image(["Nepal Bank", "Account: 1234567890"])
        files = {"file": ("test.png", img_bytes, "image/png")}
        
        response = self.client.post("/upload-document/", files=files)
        
        if response.status_code == 200:
            data = response.json()
            extracted_data = data["data"]
            
            for field, field_data in extracted_data.items():
                confidence = field_data.get("confidence", 0)
                self.assertTrue(0 <= confidence <= 1, 
                               f"Confidence for {field} out of range: {confidence}")


class TestBankDocumentScenarios(unittest.TestCase):
    """Test realistic bank document scenarios"""
    
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        
    def create_cheque_image(self):
        """Create a realistic cheque-like test image"""
        img = Image.new('RGB', (850, 400), color='white')
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Simulate cheque layout
        cheque_content = [
            (50, 50, "NEPAL BANK LIMITED"),
            (50, 100, "Pay: Ram Bahadur Sharma"),
            (50, 150, "Amount: Rs. 25,000/-"),
            (50, 200, "Date: 15/01/2024"),
            (50, 250, "A/C No: 00123456789012"),
            (400, 300, "Signature"),
        ]
        
        for x, y, text in cheque_content:
            draw.text((x, y), text, fill='black', font=font)
            
        # Add some lines to simulate cheque structure
        draw.line([(50, 170), (400, 170)], fill='black', width=1)
        draw.line([(50, 270), (400, 270)], fill='black', width=1)
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
    
    def create_deposit_slip_image(self):
        """Create a deposit slip-like test image"""
        img = Image.new('RGB', (600, 800), color='white')
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Simulate deposit slip layout
        slip_content = [
            (200, 50, "BANK DEPOSIT SLIP"),
            (50, 120, "Branch: Kathmandu"),
            (50, 170, "Date: 2024-01-15"),
            (50, 220, "Account Number: 9876543210"),
            (50, 270, "Account Name: Sita Devi"),
            (50, 320, "Cash Amount: 10000"),
            (50, 370, "Cheque Amount: 15000"),
            (50, 420, "Total: Rs. 25000"),
            (50, 500, "Depositor Name: Ram Kumar"),
            (50, 550, "Phone: 9841234567"),
        ]
        
        for x, y, text in slip_content:
            draw.text((x, y), text, fill='black', font=font)
            
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
    
    def test_process_cheque(self):
        """Test processing a cheque image"""
        cheque_image = self.create_cheque_image()
        files = {"file": ("cheque.png", cheque_image, "image/png")}
        
        response = self.client.post("/upload-document/", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        if data["success"]:
            extracted = data["data"]
            
            # Check if bank name was extracted
            if extracted["bankName"]["value"]:
                self.assertIn("BANK", extracted["bankName"]["value"].upper())
                
            # Check if account number format is valid
            if extracted["accountNumber"]["value"]:
                self.assertTrue(len(extracted["accountNumber"]["value"]) >= 10)
                
    def test_process_deposit_slip(self):
        """Test processing a deposit slip"""
        slip_image = self.create_deposit_slip_image()
        files = {"file": ("deposit_slip.png", slip_image, "image/png")}
        
        response = self.client.post("/upload-document/", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        if data["success"]:
            extracted = data["data"]
            
            # Verify structure of response
            self.assertIsInstance(extracted, dict)
            
            # Check if amount was extracted
            if extracted["amount"]["value"]:
                # Amount should contain numbers
                self.assertTrue(any(c.isdigit() for c in extracted["amount"]["value"]))
                
    def test_process_nepali_document(self):
        """Test processing document with Nepali text"""
        # Create image with Nepali text
        img = Image.new('RGB', (800, 600), color='white')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        nepali_content = [
            (50, 50, "नेपाल बैंक लिमिटेड"),
            (50, 100, "खाता नं: १२३४५६७८९०"),
            (50, 150, "नाम: राम प्रसाद शर्मा"),
            (50, 200, "रकम: रु. ५०,०००"),
            (50, 250, "मिति: २०८१/०१/०१"),
        ]
        
        for x, y, text in nepali_content:
            draw.text((x, y), text, fill='black', font=font)
            
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"file": ("nepali_doc.png", img_bytes, "image/png")}
        response = self.client.post("/upload-document/", files=files)
        
        # Should handle Nepali text without crashing
        self.assertIn(response.status_code, [200, 400])


class TestAPIErrorHandling(unittest.TestCase):
    """Test API error handling and edge cases"""
    
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        
    def test_missing_file_parameter(self):
        """Test request without file parameter"""
        response = self.client.post("/upload-document/")
        
        # Should return 422 (Unprocessable Entity) for missing required field
        self.assertEqual(response.status_code, 422)
        
    def test_multiple_file_upload(self):
        """Test uploading multiple files (should process first)"""
        img1 = self.create_simple_test_image()
        img2 = self.create_simple_test_image()
        
        # FastAPI typically handles only the first file
        files = [
            ("file", ("test1.png", img1, "image/png")),
            ("file", ("test2.png", img2, "image/png"))
        ]
        
        response = self.client.post("/upload-document/", files=files)
        
        # Should process successfully or return appropriate error
        self.assertIn(response.status_code, [200, 400, 422])
        
    def create_simple_test_image(self):
        """Helper to create a simple test image"""
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_malformed_json_response(self):
        """Test that API always returns valid JSON"""
        test_cases = [
            # Various test scenarios
            ("valid.png", self.create_simple_test_image(), "image/png"),
            ("empty.jpg", b"", "image/jpeg"),
            ("text.txt", b"text content", "text/plain"),
        ]
        
        for filename, content, content_type in test_cases:
            files = {"file": (filename, content, content_type)}
            response = self.client.post("/upload-document/", files=files)
            
            # Should always return valid JSON
            try:
                data = response.json()
                self.assertIsInstance(data, dict)
            except json.JSONDecodeError:
                self.fail(f"Response is not valid JSON for {filename}")


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBankDocumentScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)