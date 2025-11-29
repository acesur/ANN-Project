"""
Test Runner Script
==================
Comprehensive test execution with coverage reporting
"""

import unittest
import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_ocr_system import TestOCRSystem, TestBankOCRAPI, TestOCRPerformance, TestOCREdgeCases
from test_api_integration import TestAPIIntegration, TestBankDocumentScenarios, TestAPIErrorHandling
from test_with_real_images import TestRealBankDocuments, TestImageQualityScenarios


class TestRunner:
    """Comprehensive test runner with reporting"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "test_suites": [],
            "failed_tests": [],
            "errors": []
        }
        
    def run_all_tests(self):
        """Run all test suites"""
        print("="*70)
        print("BANK OCR SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*70)
        print()
        
        # Define test suites
        test_suites = [
            ("Unit Tests - OCR Core", [
                TestOCRSystem,
                TestBankOCRAPI,
                TestOCREdgeCases
            ]),
            ("Performance Tests", [
                TestOCRPerformance
            ]),
            ("Integration Tests - API", [
                TestAPIIntegration,
                TestBankDocumentScenarios,
                TestAPIErrorHandling
            ]),
            ("Real Image Tests", [
                TestRealBankDocuments,
                TestImageQualityScenarios
            ])
        ]
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_time = 0
        
        for suite_name, test_classes in test_suites:
            print(f"\n{'='*50}")
            print(f"Running: {suite_name}")
            print("="*50)
            
            suite = unittest.TestSuite()
            loader = unittest.TestLoader()
            
            for test_class in test_classes:
                suite.addTests(loader.loadTestsFromTestCase(test_class))
            
            # Run suite with timing
            start_time = time.time()
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            elapsed_time = time.time() - start_time
            
            # Collect results
            suite_results = {
                "name": suite_name,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "time": elapsed_time,
                "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
            }
            
            self.results["test_suites"].append(suite_results)
            
            # Update totals
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_time += elapsed_time
            
            # Collect failed tests
            for test, traceback in result.failures:
                self.results["failed_tests"].append({
                    "suite": suite_name,
                    "test": str(test),
                    "traceback": traceback
                })
                
            # Collect errors
            for test, traceback in result.errors:
                self.results["errors"].append({
                    "suite": suite_name,
                    "test": str(test),
                    "traceback": traceback
                })
        
        # Update summary
        self.results["summary"] = {
            "total_tests": total_tests,
            "total_failures": total_failures,
            "total_errors": total_errors,
            "total_time": total_time,
            "overall_success_rate": ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        }
        
        # Print summary
        self.print_summary()
        
        # Save results to file
        self.save_results()
        
        return total_failures == 0 and total_errors == 0
    
    def print_summary(self):
        """Print test execution summary"""
        print("\n" + "="*70)
        print("TEST EXECUTION SUMMARY")
        print("="*70)
        
        summary = self.results["summary"]
        print(f"Total Tests Run: {summary['total_tests']}")
        print(f"Failures: {summary['total_failures']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Total Time: {summary['total_time']:.2f} seconds")
        
        print("\n" + "-"*50)
        print("Test Suite Breakdown:")
        print("-"*50)
        
        for suite in self.results["test_suites"]:
            status = "✓" if suite["failures"] == 0 and suite["errors"] == 0 else "✗"
            print(f"{status} {suite['name']}")
            print(f"  Tests: {suite['tests_run']} | Failures: {suite['failures']} | Errors: {suite['errors']}")
            print(f"  Success Rate: {suite['success_rate']:.1f}% | Time: {suite['time']:.2f}s")
        
        if self.results["failed_tests"]:
            print("\n" + "-"*50)
            print("Failed Tests:")
            print("-"*50)
            for failure in self.results["failed_tests"]:
                print(f"✗ {failure['suite']} :: {failure['test']}")
        
        if self.results["errors"]:
            print("\n" + "-"*50)
            print("Tests with Errors:")
            print("-"*50)
            for error in self.results["errors"]:
                print(f"⚠ {error['suite']} :: {error['test']}")
    
    def save_results(self):
        """Save test results to JSON file"""
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"test_results_{timestamp}.json"
        
        # Remove tracebacks for JSON serialization
        json_results = self.results.copy()
        json_results["failed_tests"] = [
            {"suite": f["suite"], "test": f["test"]} 
            for f in self.results["failed_tests"]
        ]
        json_results["errors"] = [
            {"suite": e["suite"], "test": e["test"]} 
            for e in self.results["errors"]
        ]
        
        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nTest results saved to: {results_file}")


def run_specific_test_suite(suite_name):
    """Run a specific test suite"""
    suites = {
        "unit": [TestOCRSystem, TestBankOCRAPI, TestOCREdgeCases],
        "performance": [TestOCRPerformance],
        "integration": [TestAPIIntegration, TestBankDocumentScenarios, TestAPIErrorHandling],
        "real": [TestRealBankDocuments, TestImageQualityScenarios]
    }
    
    if suite_name not in suites:
        print(f"Unknown suite: {suite_name}")
        print(f"Available suites: {', '.join(suites.keys())}")
        return False
    
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    for test_class in suites[suite_name]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        suite_name = sys.argv[1].lower()
        if suite_name == "all":
            runner = TestRunner()
            success = runner.run_all_tests()
        else:
            success = run_specific_test_suite(suite_name)
    else:
        # Run all tests by default
        runner = TestRunner()
        success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)