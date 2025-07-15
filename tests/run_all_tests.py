#!python
"""
Runs all the test scripts in the tests directory using pytest.

This script:
1. Uses pytest to discover and run all test modules in the tests directory.
2. Supports running specific test modules or all tests.
3. Supports pytest's verbosity options.
4. Provides specific test categories and organization.
"""

import sys
import argparse
import logging
import pytest
import os
from pathlib import Path

# Setup logging (pytest handles its own verbose output, this is for the script itself)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_all_tests_pytest")

# Add project root to the Python path if necessary for pytest discovery
# This might not be strictly needed if tests are run from the project root
# or if pytest is configured correctly (e.g., via pytest.ini or pyproject.toml)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define test categories
TEST_CATEGORIES = {
    "all": "Run all tests",
    "md": "Run molecular dynamics tests",
    "graph": "Run graph-related tests",
    "ml": "Run machine learning tests",
    "data": "Run data processing tests"
}

def get_test_files_by_category():
    """Get test files organized by category."""
    tests_dir = Path(__file__).parent
    test_files = {
        "md": ["test_molecular_dynamics.py"],
        "graph": [
            "test_simple_graph_structure.py",
            "test_molecular_graph_structure.py",
            "test_molecular_graph_processor.py",
            "test_molecular_graph_generation.py",
            "test_hierarchical_graph_coarsening.py"
        ],
        "ml": [
            "test_mgnn_metrics.py",
            "test_mgnn_predictor.py",
            "test_mgnn_trainer.py",
            "test_hmgnn.py",
            "test_mgnn_callbacks.py",
            "test_djmgnn.py"
        ],
        "data": [
            "test_datasets.py",
            "test_dataset_loader_and_splitter.py",
            "test_data_ingestion_and_processing.py",
            "test_qm9_npz_loader.py",
            "test_pfas_feature_extraction.py"
        ]
    }
    
    # Convert to full paths
    return {
        category: [str(tests_dir / file) for file in files]
        for category, files in test_files.items()
    }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests using pytest.")
    
    # Add category argument
    parser.add_argument(
        "--category",
        "-c",
        choices=list(TEST_CATEGORIES.keys()),
        default="all",
        help="Test category to run. Default is 'all'."
    )
    
    parser.add_argument(
        "modules",
        nargs="*",
        help="Optional list of test files or directories to run (e.g., tests/test_specific_module.py or tests/). "
        'If not provided, pytest will discover tests in the current directory (usually "tests/").',
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",  # Allows for -v, -vv, etc.
        default=0,
        help="Enable verbose output (can be used multiple times for more verbosity).",
    )
    # Add any other common pytest arguments you want to expose
    parser.add_argument(
        "--pytest-args", nargs=argparse.REMAINDER, help="Additional arguments to pass directly to pytest."
    )
    return parser.parse_args()

def get_test_files(args):
    """Get the list of test files to run based on arguments."""
    if args.modules:
        return args.modules
    
    if args.category == "all":
        return None  # Let pytest discover all tests
    
    test_files_by_category = get_test_files_by_category()
    return test_files_by_category.get(args.category, [])

if __name__ == "__main__":
    args = parse_arguments()
    pytest_args = []

    # Handle verbosity
    if args.verbose > 0:
        pytest_args.append("-" + "v" * args.verbose)

    # Get test files to run
    test_files = get_test_files(args)
    if test_files:
        pytest_args.extend(test_files)
    
    # Add any extra pytest arguments
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    logger.info(f"Running tests for category: {args.category}")
    logger.info(f"Running pytest with arguments: {pytest_args}")

    # Ensure the 'tests' directory is the target if no specific modules are given
    # and the script is run from within the tests directory itself.
    # If run from project root `python tests/run_all_tests.py`, pytest will collect from `tests/`
    # If `cd tests` and then `python run_all_tests.py`, it will also collect from `.` (which is `tests/`)
    # If no specific modules are passed and pytest_args is empty,
    # explicitly add 'tests' or '.' to ensure collection from the tests directory.
    if not args.modules and not any(arg.startswith("tests") or arg == "." for arg in pytest_args):
        # If the script is in tests/ and run from tests/, '.' is fine.
        # If run from root, 'tests' is better.
        # Let's assume this script is in the 'tests' directory.
        # If no specific modules are given, pytest will search from the directory it's invoked in.
        # If this script is `tests/run_all_tests.py`, and we run `python tests/run_all_tests.py`,
        # pytest's default collection from the `tests` dir should work.
        # If we want to be explicit:
        # current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # if not args.modules and not any(arg.startswith(current_script_dir) or arg == "." for arg in pytest_args):
        # pytest_args.insert(0, current_script_dir) # Run all tests in the directory of this script
        pass

    # Execute pytest
    # The exit code from pytest.main() indicates success (0) or failure (non-zero).
    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        logger.info("All tests PASSED! ✨")
    elif exit_code == 5:
        logger.warning("No tests were collected. 🚫")
    else:
        logger.error(f"Some tests FAILED or ERRORED! ❌ (pytest exit code: {exit_code})")

    sys.exit(exit_code)
