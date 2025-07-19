"""
tests/run_all_tests.py

Runs all the test scripts in the tests directory using pytest.

This script:
1. Uses pytest to discover and run all test modules in the tests directory.
2. Supports running specific test modules or all tests.
3. Supports pytest's verbosity options.
4. Provides specific test categories and organization.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_all_tests_pytest")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

TEST_CATEGORIES: Dict[str, str] = {
    "all": "Run all tests",
    "md": "Run molecular dynamics tests",
    "graph": "Run graph-related tests",
    "ml": "Run machine learning tests",
    "data": "Run data processing tests",
}


def get_test_files_by_category() -> Dict[str, List[str]]:
    """
    Get test files organized by category.

    Returns:
        Dict[str, List[str]]: A dictionary mapping category names to lists of test file paths.
    """
    tests_dir = Path(__file__).parent
    test_files: Dict[str, List[str]] = {
        "md": ["test_molecular_dynamics.py"],
        "graph": [
            "test_molecular_graph_structure.py",
            "test_molecular_graph_processor.py",
            "test_molecular_graph_generation.py",
            "test_hierarchical_graph_coarsening.py",
        ],
        "ml": [
            "test_mgnn_metrics.py",
            "test_mgnn_predictor.py",
            "test_mgnn_trainer.py",
            "test_hmgnn.py",
            "test_mgnn_callbacks.py",
            "test_djmgnn.py",
        ],
        "data": [
            "test_datasets.py",
            "test_dataset_loader_and_splitter.py",
            "test_data_ingestion_and_processing.py",
            "test_qm9_npz_loader.py",
            "test_pfas_feature_extraction.py",
        ],
    }

    return {category: [str(tests_dir / file) for file in files] for category, files in test_files.items()}


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for running tests.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run tests using pytest.")

    parser.add_argument(
        "--category", "-c", choices=list(TEST_CATEGORIES.keys()), default="all", help="Test category to run. Default is 'all'."
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
        action="count",
        default=0,
        help="Enable verbose output (can be used multiple times for more verbosity).",
    )
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, help="Additional arguments to pass directly to pytest.")
    return parser.parse_args()


def get_test_files(args: argparse.Namespace) -> Optional[List[str]]:
    """
    Get the list of test files to run based on arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        Optional[List[str]]: A list of test file paths, or None if all tests should be run.
    """
    if args.modules:
        return args.modules

    if args.category == "all":
        return None

    test_files_by_category = get_test_files_by_category()
    return test_files_by_category.get(args.category, [])


if __name__ == "__main__":
    args = parse_arguments()
    pytest_args: List[str] = []

    if args.verbose > 0:
        pytest_args.append("-" + "v" * args.verbose)

    test_files = get_test_files(args)
    if test_files:
        pytest_args.extend(test_files)

    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    logger.info(f"Running tests for category: {args.category}")
    logger.info(f"Running pytest with arguments: {pytest_args}")

    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        logger.info("All tests PASSED! ✨")
    elif exit_code == 5:
        logger.warning("No tests were collected. 🚫")
    else:
        logger.error(f"Some tests FAILED or ERRORED! ❌ (pytest exit code: {exit_code})")

    sys.exit(exit_code)
