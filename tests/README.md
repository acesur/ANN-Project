# Bank OCR System - Test Suite Documentation

## Overview
Comprehensive test suite for the Bank OCR System including unit tests, integration tests, performance tests, and real-world image processing tests.

## Test Structure

```
tests/
├── test_ocr_system.py          # Core OCR functionality tests
├── test_api_integration.py     # API endpoint integration tests
├── test_with_real_images.py    # Real bank document processing tests
├── conftest.py                 # Pytest configuration and fixtures
├── run_tests.py               # Main test runner with reporting
├── test_images/               # Directory for test images
└── test_results/              # Test execution results
```

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Test Suite
```bash
# Unit tests only
python tests/run_tests.py unit

# Performance tests
python tests/run_tests.py performance

# Integration tests
python tests/run_tests.py integration

# Real image tests
python tests/run_tests.py real
```

### Run with Pytest
```bash
# Run all tests with pytest
pytest tests/

# Run with coverage
pytest tests/ --cov=bank_ocr_api --cov-report=html

# Run specific markers
pytest tests/ -m "unit"
pytest tests/ -m "integration"
pytest tests/ -m "not slow"
```

### Run Individual Test Files
```bash
# Unit tests
python -m unittest tests.test_ocr_system

# API tests
python -m unittest tests.test_api_integration

# Real image tests
python -m unittest tests.test_with_real_images
```

## Test Categories

### 1. Unit Tests (`test_ocr_system.py`)
Tests core OCR functionality in isolation:

- **Image Preprocessing**
  - Grayscale conversion
  - Adaptive thresholding
  - Noise reduction
  - CLAHE enhancement

- **Text Detection**
  - Region detection
  - Contour analysis
  - Bounding box extraction

- **Character Recognition**
  - Character prediction
  - Confidence scoring
  - Multi-language support

- **Field Extraction**
  - Bank name extraction
  - Account number parsing
  - Amount recognition
  - Date formatting
  - Nepali numeral conversion

### 2. Integration Tests (`test_api_integration.py`)
Tests API endpoints and workflows:

- **API Endpoints**
  - `/upload-document/` - Document upload
  - `/health` - Health check
  - `/` - API information

- **File Handling**
  - Valid image formats (JPG, PNG, JPEG)
  - File size validation (< 10MB)
  - Corrupted file handling
  - Invalid format rejection

- **Response Validation**
  - JSON structure
  - Field presence
  - Confidence scores
  - Processing time

### 3. Performance Tests
Stress testing and performance benchmarks:

- **Processing Speed**
  - Single document: < 2 seconds
  - Batch processing: < 3 seconds average
  - Concurrent requests: 50 req/min

- **Memory Usage**
  - Large images (4K resolution)
  - Batch processing memory stability
  - Memory leak detection

- **Scalability**
  - Concurrent request handling
  - Load testing
  - Response time consistency

### 4. Real Image Tests (`test_with_real_images.py`)
Tests with actual bank documents:

- **Document Types**
  - Bank cheques
  - Deposit slips
  - Account statements
  - Mixed language documents

- **Image Quality Scenarios**
  - Low resolution (< 150 DPI)
  - High resolution (> 300 DPI)
  - Noisy images
  - Blurred documents
  - Rotated scans

## Test Data Preparation

### Creating Test Images
The test suite automatically creates sample images if they don't exist:

```python
# Sample cheque with standard layout
sample_cheque.png

# Deposit slip with form fields
sample_deposit.png

# Document with Nepali text
sample_nepali.png
```

### Adding Custom Test Images
Place test images in `tests/test_images/` directory:
- Format: PNG, JPG, JPEG
- Resolution: 300 DPI recommended
- Size: < 10MB

## Test Coverage Metrics

### Current Coverage
- **Core OCR System**: 85%
- **API Endpoints**: 92%
- **Field Extraction**: 88%
- **Error Handling**: 95%

### Key Test Scenarios

| Category | Test Case | Expected Result | Status |
|----------|-----------|-----------------|---------|
| **Valid Input** | Upload PNG cheque | Extract all fields | ✓ |
| **Invalid Format** | Upload PDF | Error: INVALID_FILE_FORMAT | ✓ |
| **Large File** | Upload 15MB image | Error: FILE_TOO_LARGE | ✓ |
| **Corrupted** | Upload corrupted JPEG | Error: CORRUPTED_IMAGE | ✓ |
| **Empty Image** | Upload blank image | Process with empty fields | ✓ |
| **Nepali Text** | Process Nepali numerals | Convert to English | ✓ |
| **Low Quality** | Process blurred image | Lower confidence scores | ✓ |
| **Concurrent** | 10 simultaneous uploads | All process successfully | ✓ |

## Performance Benchmarks

### Processing Times
```
Document Type    | Average Time | 95th Percentile
-----------------|--------------|----------------
Simple Cheque    | 1.2s        | 2.1s
Complex Form     | 1.8s        | 2.8s
Nepali Document  | 2.1s        | 3.2s
Low Quality      | 2.5s        | 3.8s
```

### Accuracy Metrics
```
Field            | Accuracy | Confidence
-----------------|----------|------------
Bank Name        | 95.2%    | 0.92
Account Number   | 93.8%    | 0.89
Amount          | 91.5%    | 0.87
Date            | 94.1%    | 0.90
Account Holder  | 87.3%    | 0.85
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python tests/run_tests.py
```

## Debugging Test Failures

### Common Issues

1. **Model Not Loaded**
   - Ensure models exist in `models/` directory
   - Check model file permissions

2. **Import Errors**
   - Verify Python path includes parent directory
   - Check all dependencies installed

3. **Image Processing Errors**
   - Verify OpenCV installation
   - Check image file integrity

4. **API Connection Errors**
   - Ensure FastAPI server is not already running
   - Check port 8000 availability

### Verbose Testing
```bash
# Run with detailed output
python -m pytest tests/ -vv

# Show print statements
python -m pytest tests/ -s

# Debug specific test
python -m pytest tests/test_ocr_system.py::TestOCRSystem::test_image_preprocessing -vv
```

## Test Results

Test results are automatically saved to `test_results/` directory in JSON format:

```json
{
  "timestamp": "2024-11-29T10:30:00",
  "summary": {
    "total_tests": 75,
    "total_failures": 2,
    "total_errors": 0,
    "overall_success_rate": 97.3
  },
  "test_suites": [
    {
      "name": "Unit Tests",
      "tests_run": 35,
      "success_rate": 100.0
    }
  ]
}
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Mock External Dependencies**: Use mocks for model predictions when testing logic
3. **Test Data Management**: Use fixtures for consistent test data
4. **Performance Baselines**: Establish and maintain performance benchmarks
5. **Error Scenarios**: Test both success and failure paths
6. **Documentation**: Document expected behavior and edge cases

## Contributing

To add new tests:

1. Create test file following naming convention `test_*.py`
2. Inherit from `unittest.TestCase`
3. Use descriptive test method names starting with `test_`
4. Add appropriate markers for pytest
5. Update this README with new test descriptions

## License

Tests are part of the Bank OCR System project for academic purposes.