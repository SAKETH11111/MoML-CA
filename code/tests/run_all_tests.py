#!/usr/bin/env python3
"""
MoML-CA PFAS Project: Test Runner

This script runs all the test modules in the project and reports the overall results.
Tests are executed in sequence with clear output for each test suite.
"""

import os
import sys
import importlib.util
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_runner")

# Test modules to run
TEST_MODULES = [
    "test_smiles_validation.py",
    "test_3d_generation.py",
    "test_dataset_processing.py",
    "test_simple_graph.py",
    "test_pfas_features.py",
    "test_graph_generation.py",
    "test_data_processing.py"
]

def import_module_from_file(file_path):
    """Import a Python module from a file path."""
    module_name = os.path.basename(file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_test_module(test_file):
    """Run a test module and return whether it passed or failed."""
    print(f"\n{'='*80}")
    print(f"Running test module: {test_file}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Build the full path to the test file
    full_path = os.path.join(os.path.dirname(__file__), test_file)
    
    if not os.path.exists(full_path):
        print(f"Error: Test file {full_path} not found!")
        return False
    
    try:
        # Execute the test module as a standalone script
        result = os.system(f"{sys.executable} {full_path}")
        success = result == 0
        
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"\n✅ {test_file} PASSED in {elapsed_time:.2f} seconds")
        else:
            print(f"\n❌ {test_file} FAILED in {elapsed_time:.2f} seconds")
            
        return success
    
    except Exception as e:
        print(f"\n❌ Error running {test_file}: {str(e)}")
        return False

def run_all_tests():
    """Run all test modules and report results."""
    print("\n" + "="*40)
    print("MoML-CA PFAS Project: Test Runner")
    print("="*40 + "\n")
    
    overall_start_time = time.time()
    results = {}
    
    for test_file in TEST_MODULES:
        results[test_file] = run_test_module(test_file)
    
    # Print summary
    elapsed_time = time.time() - overall_start_time
    
    print("\n" + "="*40)
    print(f"Test Summary (total time: {elapsed_time:.2f}s)")
    print("="*40)
    
    all_passed = True
    passed_count = 0
    
    for test_file, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_file}")
        
        if passed:
            passed_count += 1
        else:
            all_passed = False
    
    print(f"\nPassed {passed_count}/{len(results)} test modules")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 All test modules passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some test modules failed. Please check the output above for details.")
        sys.exit(1) 