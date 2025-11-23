"""
Environment Setup Script for ANN Project
STW7088CEM - Artificial Neural Network

This script helps set up the Python environment and download required datasets.
Run this script before executing the Jupyter notebooks.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def create_directories():
    """Create necessary project directories"""
    directories = ['data', 'figures', 'models', 'notebooks']
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created/verified directory: {directory}/")

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} may not be compatible")
        print("  Recommended: Python 3.8 or higher")
        return False

def install_requirements():
    """Install required packages from requirements.txt"""
    print("\nInstalling required packages...")
    
    if not Path("requirements.txt").exists():
        print("✗ requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install some packages")
        return False

def verify_installations():
    """Verify that key packages are installed correctly"""
    print("\nVerifying package installations...")
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas', 
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'pytesseract': 'PyTesseract',
        'easyocr': 'EasyOCR',
        'Levenshtein': 'Python-Levenshtein',
        'nltk': 'NLTK'
    }
    
    all_good = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} is available")
        except ImportError:
            print(f"✗ {name} is not available")
            all_good = False
    
    # Special check for TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow GPU support detected: {len(gpus)} GPU(s)")
        else:
            print("ℹ TensorFlow CPU version (GPU not detected)")
    except:
        pass
    
    # Verify OpenCV functionality
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except:
        print("ℹ Could not check OpenCV version")
    
    # Check Tesseract installation
    try:
        import pytesseract
        # Try to get tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract version: {version}")
        except:
            print("ℹ Tesseract executable may not be installed (install tesseract-ocr)")
    except:
        pass
    
    return all_good

def download_sample_data():
    """Download and prepare sample datasets for testing"""
    print("\nPreparing sample datasets...")
    
    # MNIST will be downloaded automatically by TensorFlow
    print("ℹ MNIST dataset will be downloaded automatically when needed")
    
    # Create a placeholder for credit card fraud dataset
    fraud_path = Path("data/creditcard.csv")
    if not fraud_path.exists():
        print("ℹ Credit card fraud dataset not found")
        print("  Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  Save as: data/creditcard.csv")
        
        # Create a sample info file
        with open("data/dataset_info.txt", "w") as f:
            f.write("Dataset Information\n")
            f.write("==================\n\n")
            f.write("Credit Card Fraud Detection Dataset:\n")
            f.write("- Source: Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)\n")
            f.write("- File: creditcard.csv\n")
            f.write("- Size: ~150MB\n")
            f.write("- Records: 284,807 transactions\n")
            f.write("- Features: 31 (including Class label)\n")
            f.write("- Fraud rate: 0.172%\n\n")
            f.write("MNIST Dataset:\n")
            f.write("- Source: Built into TensorFlow/Keras\n")
            f.write("- Auto-downloaded on first use\n")
            f.write("- Size: ~11MB\n")
            f.write("- Images: 70,000 (60k train, 10k test)\n")
            f.write("- Classes: 10 digits (0-9)\n")
        
        print("✓ Created dataset information file")
    else:
        print("✓ Credit card fraud dataset found")

def create_jupyter_config():
    """Create Jupyter notebook configuration"""
    print("\nSetting up Jupyter configuration...")
    
    config_content = """
# Jupyter Notebook Configuration for ANN Project

# Display settings
c.InlineBackend.figure_format = 'retina'
c.InlineBackend.rc = {'figure.figsize': (10, 6)}

# Auto-reload modules
c.InteractiveShellApp.extensions = ['autoreload']
c.InteractiveShellApp.exec_lines = ['%autoreload 2']

# Increase cell width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
"""
    
    jupyter_dir = Path.home() / ".jupyter"
    jupyter_dir.mkdir(exist_ok=True)
    
    config_path = jupyter_dir / "jupyter_notebook_config.py"
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print("✓ Jupyter configuration created")

def run_basic_tests():
    """Run basic functionality tests"""
    print("\nRunning basic functionality tests...")
    
    try:
        # Test NumPy
        import numpy as np
        arr = np.array([1, 2, 3])
        assert len(arr) == 3
        print("✓ NumPy test passed")
        
        # Test Pandas
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        assert df.shape == (2, 2)
        print("✓ Pandas test passed")
        
        # Test Matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.close(fig)
        print("✓ Matplotlib test passed")
        
        # Test TensorFlow
        import tensorflow as tf
        x = tf.constant([[1.0, 2.0]])
        assert x.shape == (1, 2)
        print("✓ TensorFlow test passed")
        
        # Test Scikit-learn
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        assert X.shape == (100, 4)
        print("✓ Scikit-learn test passed")
        
        # Test OpenCV
        import cv2
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (100, 100)
        print("✓ OpenCV test passed")
        
        # Test PIL/Pillow
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        assert img.size == (100, 100)
        print("✓ Pillow test passed")
        
        # Test text processing
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, "hello", "hello").ratio()
        assert similarity == 1.0
        print("✓ Text similarity test passed")
        
        # Test Levenshtein if available
        try:
            import Levenshtein
            distance = Levenshtein.distance("hello", "hallo")
            assert distance == 1
            print("✓ Levenshtein test passed")
        except ImportError:
            print("ℹ Levenshtein not available - will use difflib")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

def setup_ocr_testing():
    """Setup OCR testing utilities"""
    print("\nSetting up OCR testing utilities...")
    
    # Create src directory if it doesn't exist
    Path("src").mkdir(exist_ok=True)
    
    # Initialize NLTK data if NLTK is available
    try:
        import nltk
        print("ℹ Downloading NLTK data (may take a few moments)...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded")
    except ImportError:
        print("ℹ NLTK not available - skipping NLTK setup")
    except Exception as e:
        print(f"ℹ NLTK setup warning: {e}")
    
    print("✓ OCR testing utilities ready")

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("SETUP COMPLETE - NEXT STEPS")
    print("="*60)
    print("\n1. Start Jupyter Notebook:")
    print("   jupyter notebook")
    print("\n2. Open notebooks in order:")
    print("   - notebooks/task1_fraud_enhanced.ipynb")
    print("   - notebooks/task2_ocr_mnist_enhanced.ipynb")
    print("   - notebooks/complete_bank_ocr_system.ipynb")
    print("   - notebooks/task3_timeseries.ipynb")
    print("   - notebooks/advanced_experiments.ipynb")
    print("\n3. For OCR testing:")
    print("   - Use: from src.ocr_testing_utils import *")
    print("   - Test images available in data/sample_document_*.jpg")
    print("   - Ground truth in data/sample_documents.json")
    print("\n4. For fraud detection:")
    print("   - Download dataset from Kaggle if not already done")
    print("   - Place creditcard.csv in data/ directory")
    print("\n5. OCR Model Testing:")
    print("   - Complete step-by-step testing methodology")
    print("   - Visual verification and error analysis")
    print("   - Comprehensive accuracy reporting")
    print("\n6. Run all cells and save results/screenshots")
    print("\n7. Check the comprehensive report:")
    print("   - docs/comprehensive_report.md")
    print("\n8. Project structure:")
    print("   - Code: In notebooks/")
    print("   - OCR Utils: src/ocr_testing_utils.py")
    print("   - Report: docs/comprehensive_report.md")
    print("   - Data: data/ (datasets and sample documents)")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("ANN Project Environment Setup")
    print("STW7088CEM - Artificial Neural Network")
    print("="*50)
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\nWarning: Python version may cause compatibility issues")
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Install requirements
    if not install_requirements():
        print("\nSetup may be incomplete due to installation issues")
        return
    
    # Step 4: Verify installations
    if not verify_installations():
        print("\nSome packages are missing - please install manually")
        return
    
    # Step 5: Download/prepare data
    download_sample_data()
    
    # Step 6: Configure Jupyter
    try:
        create_jupyter_config()
    except Exception as e:
        print(f"Warning: Could not create Jupyter config: {e}")
    
    # Step 7: Setup OCR testing
    try:
        setup_ocr_testing()
    except Exception as e:
        print(f"Warning: Could not setup OCR testing: {e}")
    
    # Step 8: Run tests
    if run_basic_tests():
        print("\n✓ All systems ready!")
    else:
        print("\n✗ Some tests failed - check your installation")
    
    # Step 9: Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()