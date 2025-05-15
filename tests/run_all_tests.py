#!python
"""
Runs all the test scripts in the tests directory using pytest.

This script:
1. Uses pytest to discover and run all test modules in the tests directory.
2. Supports running specific test modules or all tests.
3. Supports pytest's verbosity options.
"""

import sys
import argparse
import logging
import pytest
import os

# Setup logging (pytest handles its own verbose output, this is for the script itself)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_all_tests_pytest")

# Add project root to the Python path if necessary for pytest discovery
# This might not be strictly needed if tests are run from the project root
# or if pytest is configured correctly (e.g., via pytest.ini or pyproject.toml)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests using pytest.")
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


if __name__ == "__main__":
    args = parse_arguments()

    pytest_args = []

    # Handle verbosity
    if args.verbose > 0:
        pytest_args.append("-" + "v" * args.verbose)

    # Handle specific modules/paths to test
    if args.modules:
        # Pytest expects paths to files or directories.
        # The old script took module names like 'test_datasets'.
        # We'll assume the user will now provide paths like 'tests/test_datasets.py'.
        pytest_args.extend(args.modules)
    else:
        # If no modules are specified, pytest will typically search the current
        # directory or a configured testpaths directory.
        # We can explicitly tell it to run tests in the 'tests' directory.
        # This assumes the script is run from the project root or that 'tests' is discoverable.
        # If run_all_tests.py is in tests/, then pytest will discover from tests/ by default.
        pass  # Pytest default behavior is usually fine here.

    # Add any extra pytest arguments
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

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
