# Complete Bank OCR System - Deployment Guide

## System Overview
- **Models**: 3 neural networks (Character, Sequence, Detection)
- **Languages**: English + Nepali (Devanagari)
- **Purpose**: Bank document processing and field extraction

## Model Files
- `complete_ocr_character_model.h5` - Character recognition (120+ classes)
- `complete_ocr_sequence_model.h5` - Text sequence recognition
- `complete_ocr_detection_model.h5` - Text region detection
- `complete_ocr_system_metadata.json` - Character mappings and configuration

## Quick Start
```python
import tensorflow as tf
import json

# Load models
char_model = tf.keras.models.load_model('complete_ocr_character_model.h5')
seq_model = tf.keras.models.load_model('complete_ocr_sequence_model.h5')
detection_model = tf.keras.models.load_model('complete_ocr_detection_model.h5')

# Load character mappings
with open('complete_ocr_system_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
    char_mappings = metadata['character_set']['idx_to_char']

# Process new document
image = preprocess_document(your_image)  # Resize, normalize
prediction = char_model.predict(image)
character_idx = np.argmax(prediction)
recognized_char = char_mappings[str(character_idx)]
```

## System Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV 4.x
- Numpy, PIL
- For Nepali: Devanagari fonts installed

## Performance Metrics
- Character accuracy: >85% (English + Nepali combined)
- Processing speed: ~5 seconds per document
- Supported image formats: JPG, PNG
- Recommended resolution: 800x600 minimum

## Usage Notes
- Ensure good image quality (clear, well-lit)
- Install Devanagari fonts for optimal Nepali character rendering
- Use confidence thresholds (>0.8) for production filtering
- Test with your specific document templates before deployment
