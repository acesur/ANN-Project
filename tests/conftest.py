"""
Pytest Configuration and Fixtures
=================================
Shared test configuration and fixtures for all test modules
"""

import pytest
import sys
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import io
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bank_ocr_api import OCRSystem, app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def ocr_system():
    """Provide OCR system instance for all tests"""
    system = OCRSystem()
    system.load_models()
    return system


@pytest.fixture(scope="session")
def api_client():
    """Provide FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Add sample text
    cv2.putText(img, "Nepal Bank Limited", (50, 100), font, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Account: 1234567890", (50, 200), font, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Amount: Rs.50000", (50, 300), font, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Date: 2024-01-15", (50, 400), font, 0.8, (0, 0, 0), 2)
    
    return img


@pytest.fixture
def sample_cheque_image():
    """Create a sample cheque image"""
    img = np.ones((400, 850, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Bank name
    cv2.putText(img, "NEPAL BANK LIMITED", (50, 50), font, 1.0, (0, 0, 0), 2)
    
    # Pay line
    cv2.putText(img, "Pay: Mr. Ram Sharma", (50, 120), font, 0.7, (0, 0, 0), 1)
    
    # Amount
    cv2.putText(img, "Rs. 25,000/-", (610, 180), font, 0.8, (0, 0, 0), 2)
    
    # Account number
    cv2.putText(img, "A/C No: 00123456789012", (50, 250), font, 0.6, (0, 0, 0), 1)
    
    # Date
    cv2.putText(img, "Date: 15/01/2024", (600, 50), font, 0.6, (0, 0, 0), 1)
    
    return img


@pytest.fixture
def image_as_bytes(sample_image):
    """Convert numpy image to bytes for API upload"""
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
    
    # Save to bytes
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_ocr_response():
    """Mock OCR response for testing"""
    return {
        "bankName": {"value": "Test Bank", "confidence": 0.95},
        "accountHolderName": {"value": "John Doe", "confidence": 0.88},
        "accountNumber": {"value": "1234567890", "confidence": 0.92},
        "routingNumber": {"value": "987654321", "confidence": 0.85},
        "amount": {"value": "50000", "confidence": 0.90},
        "date": {"value": "2024-01-15", "confidence": 0.93}
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require trained models"
    )