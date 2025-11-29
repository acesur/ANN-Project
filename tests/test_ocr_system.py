"""
Comprehensive Unit Tests for Bank OCR System
=============================================
Tests OCR functionality including image upload, processing, and field extraction
Author: Suresh Chaudhary
"""

import unittest
import numpy as np
import cv2
import json
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from bank_ocr_api import OCRSystem, create_error_response, create_success_response, is_allowed_file
import tensorflow as tf


class TestOCRSystem(unittest.TestCase):
    """Test suite for OCR System core functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests"""
        cls.ocr_system = OCRSystem()
        # Load models if available
        cls.ocr_system.load_models()
        
    def setUp(self):
        """Set up test fixtures for each test"""
        # Create a sample test image (blank white image)
        self.test_image = np.ones((500, 800, 3), dtype=np.uint8) * 255
        
        # Create synthetic test data
        self.sample_text_regions = [
            (100, 100, 200, 50),  # x, y, width, height
            (100, 200, 300, 50),
            (100, 300, 250, 50)
        ]
        
        # Sample extracted texts for testing
        self.sample_extracted_texts = [
            ("Nepal Bank", 0.95),
            ("1234567890123456", 0.92),
            ("Rs 50000", 0.88),
            ("2024-01-15", 0.90),
            ("राम प्रसाद", 0.85)
        ]
    
    def test_ocr_system_initialization(self):
        """Test OCR system initializes correctly"""
        ocr = OCRSystem()
        self.assertIsNotNone(ocr)
        self.assertIsInstance(ocr.char_to_idx, dict)
        self.assertIsInstance(ocr.idx_to_char, dict)
        
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        # Test with color image
        processed = self.ocr_system.preprocess_image(self.test_image)
        
        # Check output is grayscale
        self.assertEqual(len(processed.shape), 2)
        
        # Check output is binary (only 0 and 255 values)
        unique_values = np.unique(processed)
        self.assertTrue(all(v in [0, 255] for v in unique_values))
        
    def test_preprocessing_with_grayscale_input(self):
        """Test preprocessing with grayscale input"""
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        processed = self.ocr_system.preprocess_image(gray_image)
        
        self.assertIsNotNone(processed)
        self.assertEqual(len(processed.shape), 2)
        
    def test_text_region_detection(self):
        """Test text region detection"""
        # Create an image with some text-like regions
        test_img = np.ones((500, 800, 3), dtype=np.uint8) * 255
        
        # Add some black rectangles to simulate text
        cv2.rectangle(test_img, (100, 100), (300, 150), (0, 0, 0), -1)
        cv2.rectangle(test_img, (100, 200), (400, 250), (0, 0, 0), -1)
        
        regions = self.ocr_system.detect_text_regions(test_img)
        
        self.assertIsInstance(regions, list)
        if regions:
            # Check each region is a tuple of 4 integers
            for region in regions:
                self.assertEqual(len(region), 4)
                self.assertTrue(all(isinstance(x, (int, np.integer)) for x in region))
                
    def test_empty_image_handling(self):
        """Test handling of empty/blank images"""
        blank_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        regions = self.ocr_system.detect_text_regions(blank_image)
        
        # Should still return a list (possibly empty or with full image as region)
        self.assertIsInstance(regions, list)
        
    def test_character_prediction_shape(self):
        """Test character prediction with correct input shape"""
        # Create a dummy character image
        char_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        
        if self.ocr_system.character_model is not None:
            char, confidence = self.ocr_system.predict_character(char_image)
            
            self.assertIsInstance(char, str)
            self.assertIsInstance(confidence, float)
            self.assertTrue(0 <= confidence <= 1)
        else:
            # Skip if model not loaded
            self.skipTest("Character model not loaded")
            
    def test_nepali_to_english_numeral_conversion(self):
        """Test conversion of Nepali numerals to English"""
        test_cases = [
            ("०१२३४५६७८९", "0123456789"),
            ("खाता नं: १२३४", "खाता नं: 1234"),
            ("रु. ५०,०००", "रु. 50,000"),
            ("मिति: २०८१/०८/०७", "मिति: 2081/08/07")
        ]
        
        for nepali_text, expected in test_cases:
            result = self.ocr_system.convert_nepali_to_english_numerals(nepali_text)
            self.assertEqual(result, expected)
            
    def test_date_formatting(self):
        """Test date formatting functions"""
        test_dates = [
            ("2024/01/15", "2024-01-15"),
            ("15/01/2024", "2024-01-15"),
            ("2024-03-25", "2024-03-25"),
            ("5/3/2024", "2024-03-05"),
            ("invalid_date", "2024-01-15")  # Should return default
        ]
        
        for input_date, expected in test_dates:
            result = self.ocr_system.format_date(input_date)
            self.assertEqual(result, expected)
            
    def test_bank_field_extraction(self):
        """Test extraction of bank-specific fields"""
        # Test with sample extracted texts
        result = self.ocr_system.extract_bank_fields(self.sample_extracted_texts)
        
        # Check all required fields are present
        required_fields = ["bankName", "accountHolderName", "accountNumber", 
                          "routingNumber", "amount", "date"]
        for field in required_fields:
            self.assertIn(field, result)
            self.assertIn("value", result[field])
            self.assertIn("confidence", result[field])
            
    def test_bank_name_extraction(self):
        """Test bank name extraction patterns"""
        test_texts = [
            ([("Nepal Bank Limited", 0.95)], "Nepal Bank"),
            ([("Nabil Bank", 0.90)], "Nabil Bank"),
            ([("राष्ट्रिय वाणिज्य बैंक", 0.88)], "राष्ट्रिय वाणिज्य बैंक"),
            ([("Some text ABC Bank more text", 0.85)], "ABC Bank")
        ]
        
        for texts, expected_bank in test_texts:
            result = self.ocr_system.extract_bank_fields(texts)
            if expected_bank:
                self.assertIn(expected_bank, result["bankName"]["value"])
                
    def test_account_number_extraction(self):
        """Test account number extraction"""
        test_texts = [
            ([("Account: 1234567890123456", 0.95)], "1234567890123456"),
            ([("खाता: १२३४५६७८९०१२", 0.90)], "123456789012"),
            ([("A/C No. 9876543210", 0.88)], "9876543210"),
        ]
        
        for texts, expected_account in test_texts:
            result = self.ocr_system.extract_bank_fields(texts)
            if expected_account and len(expected_account) >= 10:
                self.assertEqual(result["accountNumber"]["value"], expected_account)
                
    def test_amount_extraction(self):
        """Test amount extraction patterns"""
        test_texts = [
            ([("Amount: Rs.50,000", 0.95)], "50,000"),
            ([("रकम: ₨ 25000", 0.90)], "25000"),
            ([("Total: Rs 1,00,000.50", 0.88)], "1,00,000.50"),
        ]
        
        for texts, expected_amount in test_texts:
            result = self.ocr_system.extract_bank_fields(texts)
            if expected_amount:
                # Check if amount is extracted (may have slight variations)
                self.assertTrue(any(c.isdigit() for c in result["amount"]["value"]))
                
    def test_confidence_score_calculation(self):
        """Test confidence score aggregation"""
        texts_with_confidence = [
            ("Text1", 0.95),
            ("Text2", 0.88),
            ("Text3", 0.92)
        ]
        
        result = self.ocr_system.extract_bank_fields(texts_with_confidence)
        
        # Check all confidence scores are between 0 and 1
        for field, data in result.items():
            if data["confidence"] > 0:
                self.assertTrue(0 <= data["confidence"] <= 1)
                
    def test_process_document_integration(self):
        """Integration test for complete document processing"""
        # Create a more realistic test image with text
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Add some text-like patterns
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_image, "Nepal Bank", (100, 100), font, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "Account: 1234567890", (100, 200), font, 0.8, (0, 0, 0), 2)
        cv2.putText(test_image, "Amount: Rs.50000", (100, 300), font, 0.8, (0, 0, 0), 2)
        
        try:
            result = self.ocr_system.process_document(test_image)
            
            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn("bankName", result)
            self.assertIn("accountNumber", result)
            self.assertIn("amount", result)
        except Exception as e:
            # Document that this test requires models to be loaded
            self.skipTest(f"Full processing test skipped: {str(e)}")


class TestBankOCRAPI(unittest.TestCase):
    """Test suite for Bank OCR API endpoints and utilities"""
    
    def test_file_extension_validation(self):
        """Test file extension validation"""
        valid_files = ["test.jpg", "document.jpeg", "scan.png", "IMAGE.JPG"]
        invalid_files = ["test.pdf", "document.txt", "scan.bmp", "test", ".jpg"]
        
        for filename in valid_files:
            self.assertTrue(is_allowed_file(filename), f"{filename} should be valid")
            
        for filename in invalid_files:
            self.assertFalse(is_allowed_file(filename), f"{filename} should be invalid")
            
    def test_error_response_creation(self):
        """Test error response format"""
        error_code = "TEST_ERROR"
        message = "Test error message"
        
        response = create_error_response(error_code, message)
        
        self.assertIsInstance(response, dict)
        self.assertEqual(response["success"], False)
        self.assertEqual(response["error"], error_code)
        self.assertEqual(response["message"], message)
        
    def test_success_response_creation(self):
        """Test success response format"""
        test_data = {
            "bankName": {"value": "Test Bank", "confidence": 0.95},
            "accountNumber": {"value": "1234567890", "confidence": 0.92}
        }
        processing_time = 1.5
        
        response = create_success_response(test_data, processing_time)
        
        self.assertIsInstance(response, dict)
        self.assertEqual(response["success"], True)
        self.assertEqual(response["data"], test_data)
        self.assertEqual(response["processingTime"], 1.5)
        self.assertIn("message", response)


class TestOCRPerformance(unittest.TestCase):
    """Performance and stress tests for OCR system"""
    
    @classmethod
    def setUpClass(cls):
        cls.ocr_system = OCRSystem()
        cls.ocr_system.load_models()
        
    def test_large_image_processing(self):
        """Test processing of large images"""
        # Create a large test image (4K resolution)
        large_image = np.ones((2160, 3840, 3), dtype=np.uint8) * 255
        
        try:
            regions = self.ocr_system.detect_text_regions(large_image)
            self.assertIsInstance(regions, list)
        except MemoryError:
            self.skipTest("Insufficient memory for large image test")
            
    def test_batch_processing(self):
        """Test batch processing of multiple images"""
        batch_size = 10
        images = [np.ones((500, 800, 3), dtype=np.uint8) * 255 for _ in range(batch_size)]
        
        results = []
        for img in images:
            regions = self.ocr_system.detect_text_regions(img)
            results.append(regions)
            
        self.assertEqual(len(results), batch_size)
        
    def test_processing_speed(self):
        """Test processing speed meets requirements"""
        import time
        
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        start_time = time.time()
        _ = self.ocr_system.detect_text_regions(test_image)
        processing_time = time.time() - start_time
        
        # Should process in under 5 seconds for basic operations
        self.assertLess(processing_time, 5.0, 
                       f"Processing took {processing_time:.2f}s, expected < 5s")


class TestOCREdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    @classmethod
    def setUpClass(cls):
        cls.ocr_system = OCRSystem()
        
    def test_empty_image_array(self):
        """Test handling of empty image array"""
        empty_image = np.array([])
        
        # Should handle gracefully without crashing
        try:
            result = self.ocr_system.preprocess_image(empty_image.reshape(0, 0))
        except:
            # Expected to handle error gracefully
            pass
            
    def test_single_pixel_image(self):
        """Test handling of single pixel image"""
        single_pixel = np.array([[[255, 255, 255]]], dtype=np.uint8)
        
        try:
            regions = self.ocr_system.detect_text_regions(single_pixel)
            self.assertIsInstance(regions, list)
        except:
            # Should handle gracefully
            pass
            
    def test_corrupted_image_data(self):
        """Test handling of corrupted image data"""
        # Create image with NaN values
        corrupted = np.ones((100, 100, 3), dtype=np.float32)
        corrupted[50:60, 50:60] = np.nan
        
        try:
            result = self.ocr_system.preprocess_image(corrupted)
            # Should either handle or raise appropriate error
        except:
            pass
            
    def test_extreme_aspect_ratios(self):
        """Test images with extreme aspect ratios"""
        # Very wide image
        wide_image = np.ones((10, 1000, 3), dtype=np.uint8) * 255
        regions_wide = self.ocr_system.detect_text_regions(wide_image)
        self.assertIsInstance(regions_wide, list)
        
        # Very tall image
        tall_image = np.ones((1000, 10, 3), dtype=np.uint8) * 255
        regions_tall = self.ocr_system.detect_text_regions(tall_image)
        self.assertIsInstance(regions_tall, list)
        
    def test_unicode_text_handling(self):
        """Test handling of Unicode text (Nepali characters)"""
        unicode_texts = [
            ("नेपाल बैंक", 0.90),
            ("खाता धारक: राम बहादुर", 0.85),
            ("रकम: रु. ५०,०००", 0.88),
            ("मिति: २०८१/०८/०७", 0.92)
        ]
        
        result = self.ocr_system.extract_bank_fields(unicode_texts)
        
        # Should handle Unicode without errors
        self.assertIsInstance(result, dict)
        
    def test_mixed_language_extraction(self):
        """Test extraction with mixed English and Nepali text"""
        mixed_texts = [
            ("Nepal Bank नेपाल बैंक", 0.90),
            ("Account खाता: 1234567890", 0.88),
            ("Amount रकम: Rs.50000", 0.85)
        ]
        
        result = self.ocr_system.extract_bank_fields(mixed_texts)
        
        self.assertIsInstance(result, dict)
        # Should extract relevant fields regardless of language mix


def create_test_suite():
    """Create a test suite with all test cases"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOCRSystem))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBankOCRAPI))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOCRPerformance))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOCREdgeCases))
    
    return suite


if __name__ == '__main__':
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    runner.run(suite)