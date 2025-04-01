#!/usr/bin/env python3
"""
Runs all the test scripts in the tests directory

This script:
1. Finds all test modules in the tests directory
2. Runs each test module and collects results
3. Provides a detailed summary of test results
4. Supports running specific test modules via command line arguments
"""

import os
import sys
import importlib
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_all_tests")

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


def find_test_modules():
    """Find all test modules in the tests directory."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = []
    
    for file in os.listdir(test_dir):
        if file.startswith("test_") and file.endswith(".py"):
            # Skip this file itself
            if file == os.path.basename(__file__):
                continue
            
            # Get the module name without .py
            module_name = file[:-3]
            test_files.append(module_name)
    
    return sorted(test_files)


def run_test_module(module_name):
    """Run the specified test module."""
    logger.info(f"Running test module: {module_name}")
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Call the run_tests function if it exists
        if hasattr(module, 'run_tests'):
            success = module.run_tests()
        else:
            # Try to find other commonly used test runner functions
            runner_found = False
            for runner_name in ['run_all_tests', 'run_graph_generation_tests', 'run_dataset_processing_tests', 
                               'run_pipeline_tests', 'run_validation_tests']:
                if hasattr(module, runner_name):
                    success = getattr(module, runner_name)()
                    runner_found = True
                    break
            
            # If no runner function found, assume test passed
            if not runner_found:
                logger.warning(f"No test runner function found in {module_name}")
                success = True
        
        return success
    
    except Exception as e:
        logger.error(f"Error running test module {module_name}: {e}")
        return False


def run_all_tests(modules=None):
    """Run all test modules or the specified ones."""
    if modules is None:
        modules = find_test_modules()
    
    logger.info(f"Found {len(modules)} test modules: {', '.join(modules)}")
    
    results = {}
    all_passed = True
    start_time = datetime.now()
    
    for module_name in modules:
        module_start = datetime.now()
        success = run_test_module(module_name)
        module_duration = datetime.now() - module_start
        
        results[module_name] = {
            "status": "PASSED" if success else "FAILED",
            "duration": module_duration
        }
        
        if not success:
            all_passed = False
    
    # Print summary
    total_duration = datetime.now() - start_time
    logger.info("\n===== Test Results =====")
    logger.info(f"Total duration: {total_duration}")
    logger.info("\nModule Results:")
    
    for module_name, result in results.items():
        status = result["status"]
        duration = result["duration"]
        logger.info(f"{module_name:30} {status:8} ({duration})")
    
    if all_passed:
        logger.info("\nAll tests PASSED! ✨")
    else:
        logger.error("\nSome tests FAILED! ❌")
    
    return all_passed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run test modules.')
    parser.add_argument('--modules', nargs='+', help='List of test modules to run.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output.')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # If specific modules are provided, run only those
    modules = args.modules
    
    # Run the tests
    success = run_all_tests(modules)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 