#!/usr/bin/env python3
"""
Test runner for MoML-CA

This script runs all tests in the codebase to ensure everything is working correctly.
It integrates with pytest to provide a comprehensive test report.

Run with: python run_tests.py
"""

import os
import sys
import subprocess
import argparse
import time

def run_module_tests(module_path, verbose=False):
    """Run tests for a specific module."""
    cmd = ["python", "-m", "pytest", module_path]
    if verbose:
        cmd.append("-v")
    
    print(f"\n{'='*80}")
    print(f"Running tests for: {module_path}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    print(f"\nCompleted in {elapsed_time:.2f} seconds with exit code {result.returncode}")
    return result.returncode

def run_all_tests(verbose=False):
    """Run all tests in the codebase."""
    print("\n🧪 Running all MoML-CA tests...")
    
    test_modules = [
        "code/tests",
        "code/MGNN/tests",
        "code/integration/orchestration/test_pipeline.py"
    ]
    
    results = {}
    all_passed = True
    
    for module in test_modules:
        returncode = run_module_tests(module, verbose)
        results[module] = returncode == 0
        if returncode != 0:
            all_passed = False
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for module, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{module:<50} {status}")
    
    # Final verdict
    if all_passed:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
        return 1

def run_specific_test(test_path, verbose=False):
    """Run a specific test file or directory."""
    print(f"\n🧪 Running specific test: {test_path}")
    return run_module_tests(test_path, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MoML-CA tests")
    parser.add_argument("--test", type=str, help="Specific test file or directory to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    if args.test:
        sys.exit(run_specific_test(args.test, args.verbose))
    else:
        sys.exit(run_all_tests(args.verbose)) 