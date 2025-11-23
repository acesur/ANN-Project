"""
OCR Testing Utilities for ANN Project
STW7088CEM - Artificial Neural Network

This module provides comprehensive testing utilities for OCR model validation.
"""

import cv2
import json
import numpy as np
from PIL import Image
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import re

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

def preprocess_image_for_ocr(image_path: str) -> np.ndarray:
    """
    Preprocess image for OCR analysis
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def extract_numbers_from_text(text: str) -> str:
    """Extract numeric characters from text"""
    return ''.join(re.findall(r'\d', text))

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    # Use Levenshtein if available, otherwise use difflib
    if LEVENSHTEIN_AVAILABLE and text1 and text2:
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        distance = Levenshtein.distance(text1, text2)
        return 1 - (distance / max_len)
    else:
        return SequenceMatcher(None, text1, text2).ratio()

def calculate_field_accuracy(predicted: Dict[str, str], ground_truth: Dict[str, str]) -> Dict[str, Any]:
    """
    Calculate accuracy metrics for extracted fields
    
    Args:
        predicted: Predicted field values
        ground_truth: True field values
        
    Returns:
        Dictionary with accuracy metrics
    """
    metrics = {}
    
    # Standard fields to evaluate
    text_fields = ['bank_name', 'account_holder']
    numeric_fields = ['account_number', 'routing_number', 'amount']
    
    # Exact match accuracy for all fields
    for field in text_fields + numeric_fields:
        pred_value = predicted.get(field, '').strip()
        true_value = ground_truth.get(field, '').strip()
        metrics[f'{field}_exact_match'] = pred_value == true_value
    
    # Text similarity for text fields
    for field in text_fields:
        pred_value = predicted.get(field, '').strip()
        true_value = ground_truth.get(field, '').strip()
        similarity = calculate_text_similarity(pred_value, true_value)
        metrics[f'{field}_similarity'] = similarity
        metrics[f'{field}_high_similarity'] = similarity >= 0.8
    
    # Numeric accuracy for numeric fields
    for field in numeric_fields:
        pred_value = predicted.get(field, '')
        true_value = ground_truth.get(field, '')
        
        pred_numbers = extract_numbers_from_text(pred_value)
        true_numbers = extract_numbers_from_text(true_value)
        
        metrics[f'{field}_numeric_match'] = pred_numbers == true_numbers
        
        # Partial numeric similarity
        if pred_numbers and true_numbers:
            numeric_similarity = calculate_text_similarity(pred_numbers, true_numbers)
            metrics[f'{field}_numeric_similarity'] = numeric_similarity
        else:
            metrics[f'{field}_numeric_similarity'] = 0.0
    
    return metrics

def test_ocr_model_accuracy(model, test_images: List[str], ground_truth: List[Dict], 
                          preprocessing_func=None, prediction_func=None, 
                          field_extraction_func=None) -> List[Dict]:
    """
    Test OCR model accuracy on a set of test images
    
    Args:
        model: Trained OCR model
        test_images: List of image paths
        ground_truth: List of ground truth dictionaries
        preprocessing_func: Custom preprocessing function
        prediction_func: Custom prediction function
        field_extraction_func: Custom field extraction function
        
    Returns:
        List of test results
    """
    results = []
    
    for i, img_path in enumerate(test_images):
        print(f"Testing image {i+1}/{len(test_images)}: {img_path}")
        
        try:
            # Preprocess image
            if preprocessing_func:
                processed_img = preprocessing_func(img_path)
            else:
                processed_img = preprocess_image_for_ocr(img_path)
            
            # Make prediction (this needs to be customized for your model)
            if prediction_func:
                predictions = prediction_func(model, processed_img)
            else:
                # Default prediction - needs to be implemented based on your model
                predictions = {"error": "No prediction function provided"}
            
            # Extract fields (this needs to be customized for your model output)
            if field_extraction_func:
                extracted_fields = field_extraction_func(predictions)
            else:
                # Default extraction - needs to be implemented
                extracted_fields = {"error": "No field extraction function provided"}
            
            # Calculate accuracy if we have ground truth
            if i < len(ground_truth):
                gt = ground_truth[i]
                accuracy_scores = calculate_field_accuracy(extracted_fields, gt)
            else:
                accuracy_scores = {}
            
            result = {
                'image_path': img_path,
                'predicted_fields': extracted_fields,
                'ground_truth': ground_truth[i] if i < len(ground_truth) else {},
                'accuracy_metrics': accuracy_scores,
                'processing_success': True
            }
            
        except Exception as e:
            result = {
                'image_path': img_path,
                'error': str(e),
                'processing_success': False
            }
        
        results.append(result)
    
    return results

def generate_accuracy_report(test_results: List[Dict]) -> Dict[str, Any]:
    """
    Generate comprehensive accuracy report from test results
    
    Args:
        test_results: List of test result dictionaries
        
    Returns:
        Comprehensive accuracy report
    """
    successful_tests = [r for r in test_results if r.get('processing_success', False)]
    total_tests = len(test_results)
    successful_count = len(successful_tests)
    
    if successful_count == 0:
        return {
            'error': 'No successful tests to analyze',
            'total_tests': total_tests,
            'successful_tests': 0
        }
    
    # Calculate field-wise accuracies
    field_accuracies = {}
    fields = ['bank_name', 'account_holder', 'account_number', 'routing_number', 'amount']
    
    for field in fields:
        exact_matches = sum([
            r['accuracy_metrics'].get(f'{field}_exact_match', False) 
            for r in successful_tests
        ])
        field_accuracies[f'{field}_exact'] = (exact_matches / successful_count) * 100
        
        # Text similarity for text fields
        if field in ['bank_name', 'account_holder']:
            similarities = [
                r['accuracy_metrics'].get(f'{field}_similarity', 0) 
                for r in successful_tests
            ]
            field_accuracies[f'{field}_avg_similarity'] = np.mean(similarities) * 100
            
            high_sim_count = sum([
                r['accuracy_metrics'].get(f'{field}_high_similarity', False) 
                for r in successful_tests
            ])
            field_accuracies[f'{field}_high_similarity'] = (high_sim_count / successful_count) * 100
        
        # Numeric similarity for numeric fields
        if field in ['account_number', 'routing_number', 'amount']:
            numeric_matches = sum([
                r['accuracy_metrics'].get(f'{field}_numeric_match', False) 
                for r in successful_tests
            ])
            field_accuracies[f'{field}_numeric'] = (numeric_matches / successful_count) * 100
    
    # Overall document accuracy (all fields correct)
    perfect_documents = sum([
        all([
            r['accuracy_metrics'].get(f'{field}_exact_match', False) 
            for field in fields
        ])
        for r in successful_tests
    ])
    
    report = {
        'total_tests': total_tests,
        'successful_tests': successful_count,
        'failed_tests': total_tests - successful_count,
        'overall_success_rate': (successful_count / total_tests) * 100,
        'perfect_document_accuracy': (perfect_documents / successful_count) * 100,
        'field_accuracies': field_accuracies,
        'perfect_documents': perfect_documents
    }
    
    return report

def visualize_test_results(test_results: List[Dict], save_path: str = None):
    """
    Create visualizations of test results
    
    Args:
        test_results: List of test result dictionaries
        save_path: Optional path to save the visualization
    """
    successful_tests = [r for r in test_results if r.get('processing_success', False)]
    
    if not successful_tests:
        print("No successful tests to visualize")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('OCR Model Test Results', fontsize=16, fontweight='bold')
    
    # Field accuracy comparison
    fields = ['bank_name', 'account_holder', 'account_number', 'routing_number', 'amount']
    exact_accuracies = []
    
    for field in fields:
        exact_matches = sum([
            r['accuracy_metrics'].get(f'{field}_exact_match', False) 
            for r in successful_tests
        ])
        exact_accuracies.append((exact_matches / len(successful_tests)) * 100)
    
    axes[0, 0].bar(fields, exact_accuracies, color='skyblue', edgecolor='navy')
    axes[0, 0].set_title('Field Exact Match Accuracy (%)')
    axes[0, 0].set_ylabel('Accuracy %')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Text similarity for name fields
    name_fields = ['bank_name', 'account_holder']
    similarities = {field: [] for field in name_fields}
    
    for field in name_fields:
        for result in successful_tests:
            sim = result['accuracy_metrics'].get(f'{field}_similarity', 0)
            similarities[field].append(sim * 100)
    
    axes[0, 1].boxplot([similarities[field] for field in name_fields], 
                       labels=name_fields)
    axes[0, 1].set_title('Text Similarity Distribution (%)')
    axes[0, 1].set_ylabel('Similarity %')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Overall success rate
    total_tests = len(test_results)
    success_rate = len(successful_tests) / total_tests * 100
    failure_rate = 100 - success_rate
    
    axes[1, 0].pie([success_rate, failure_rate], 
                   labels=['Successful Tests', 'Failed Tests'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%')
    axes[1, 0].set_title('Test Success Rate')
    
    # Perfect document accuracy
    perfect_docs = sum([
        all([
            r['accuracy_metrics'].get(f'{field}_exact_match', False) 
            for field in fields
        ])
        for r in successful_tests
    ])
    perfect_rate = (perfect_docs / len(successful_tests)) * 100
    partial_rate = 100 - perfect_rate
    
    axes[1, 1].pie([perfect_rate, partial_rate],
                   labels=['Perfect Documents', 'Partial Accuracy'],
                   colors=['gold', 'lightblue'],
                   autopct='%1.1f%%')
    axes[1, 1].set_title('Document-level Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def save_test_results(test_results: List[Dict], accuracy_report: Dict[str, Any], 
                     results_path: str = 'ocr_test_results.json',
                     report_path: str = 'ocr_accuracy_report.json'):
    """
    Save test results and accuracy report to JSON files
    
    Args:
        test_results: List of test result dictionaries
        accuracy_report: Accuracy report dictionary
        results_path: Path to save detailed results
        report_path: Path to save accuracy report
    """
    # Save detailed results
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
    
    # Save accuracy report
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(accuracy_report, f, indent=2, default=str)
    
    print(f"✓ Test results saved to: {results_path}")
    print(f"✓ Accuracy report saved to: {report_path}")

if __name__ == "__main__":
    print("OCR Testing Utilities Module")
    print("Import this module to use OCR testing functions")