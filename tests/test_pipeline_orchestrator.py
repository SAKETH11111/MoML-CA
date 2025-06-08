import os
import sys
import time
import tempfile
import unittest
import shutil
import pandas as pd
import logging
from typing import Dict, Tuple

#!python
"""
Test script for the PFAS Pipeline Orchestrator

This script tests the optimized pipeline stages, verifying that:
1. Data preprocessing works correctly
2. ORCA calculation orchestration functions properly (mocked if ORCA not available)
3. Molecular graph generation succeeds
4. Checkpointing and caching mechanisms function as expected
5. The resume functionality recovers from interruptions

Run with: python -m code.integration.orchestration.test_pipeline
"""


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline_test")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pipeline components
try:
    from moml.pipeline import PFASPipelineOrchestrator

    PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Pipeline module not available: {e}")
    PIPELINE_AVAILABLE = False

# Sample SMILES for testing
TEST_SMILES = [
    # Valid PFAS SMILES
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFOA"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFHxA"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFBA"),
    # This SMILES is actually considered valid by RDKit, but has missing parentheses
    ("CC(CC(C)(C))", "SYNTAX_ERROR"),
    # Invalid SMILES for testing error handling
    ("INVALID_SMILES", "INVALID"),
]


def create_test_dataset(output_dir: str) -> str:
    """Create a test dataset with PFAS SMILES strings."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a DataFrame with SMILES and names
    data = {"SMILES": [smile[0] for smile in TEST_SMILES], "common_name": [smile[1] for smile in TEST_SMILES]}
    df = pd.DataFrame(data)

    # Save to CSV
    output_file = os.path.join(output_dir, "test_pfas_dataset.csv")
    df.to_csv(output_file, index=False)

    return output_file


def create_test_config(base_dir: str) -> Dict:
    """Create a test configuration for the pipeline."""
    return {
        "data_dir": os.path.join(base_dir, "data"),
        "output_dir": os.path.join(base_dir, "output"),
        "working_dir": os.path.join(base_dir, "working"),
        "parallel": {"enabled": True, "max_workers": 2},
        "qm": {"functional": "B3LYP", "basis_set": "6-31G*", "num_procs": 1, "memory": 1000},
        "graph": {"charge_type": "mulliken", "use_pfas_features": True, "use_quantum_properties": True},
        "execution": {
            "skip_qm": False,
            "skip_graph_generation": False,
            "force_rerun": False,
            "cache_intermediates": True,
        },
    }


def setup_test_environment() -> Tuple[str, str, Dict]:
    """Set up a test environment with temporary directories."""
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="moml_ca_test_")

    # Set up subdirectories
    os.makedirs(os.path.join(test_dir, "data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "working"), exist_ok=True)

    # Create test dataset
    test_dataset = create_test_dataset(os.path.join(test_dir, "data", "raw"))

    # Create test configuration
    test_config = create_test_config(test_dir)

    return test_dir, test_dataset, test_config


def cleanup_test_environment(test_dir: str):
    """Clean up the test environment."""
    try:
        shutil.rmtree(test_dir)
        logger.info(f"Cleaned up test directory: {test_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up test directory {test_dir}: {e}")


class TestPFASPipelineOrchestrator(unittest.TestCase):
    """Test case for the PFAS Pipeline Orchestrator."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir, cls.test_dataset, cls.test_config = setup_test_environment()
        logger.info(f"Set up test environment in {cls.test_dir}")

        # Initialize orchestrator with test configuration
        cls.orchestrator = PFASPipelineOrchestrator(
            data_dir=cls.test_config["data_dir"],
            output_dir=cls.test_config["output_dir"],
            working_dir=cls.test_config["working_dir"],
            # cache_intermediates is handled by the config dictionary
        )

        # Update configuration
        cls.orchestrator.config.update(cls.test_config)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        cleanup_test_environment(cls.test_dir)

    def test_01_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.config["data_dir"], self.test_config["data_dir"])
        self.assertEqual(self.orchestrator.config["output_dir"], self.test_config["output_dir"])
        self.assertEqual(self.orchestrator.config["working_dir"], self.test_config["working_dir"])

        # Check directory creation
        self.assertTrue(os.path.exists(self.orchestrator.dirs["raw_data"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["processed_data"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["orca_input"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["orca_output"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["molecule_files"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["molecular_graphs"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["checkpoints"]))

    def test_02_preprocess_data(self):
        """Test preprocessing stage."""
        logger.info("Testing preprocessing stage...")

        # Run preprocessing
        df = self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=True)

        # Verify results
        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(TEST_SMILES))
        self.assertTrue("is_valid_smiles" in df.columns)
        self.assertTrue("canonical_smiles" in df.columns)
        self.assertTrue("molecular_weight" in df.columns)

        # Check valid/invalid SMILES counts
        valid_count = sum(df["is_valid_smiles"])
        invalid_count = len(df) - valid_count
        self.assertEqual(valid_count, 4)  # We have 4 valid SMILES in TEST_SMILES
        self.assertEqual(invalid_count, 1)  # We have 1 invalid SMILES in TEST_SMILES

        # Check output file - construct the expected filename dynamically
        base_input_filename = os.path.splitext(os.path.basename(self.test_dataset))[0]
        expected_output_filename = f"{base_input_filename}_pfas_processed.csv"
        output_file = os.path.join(self.orchestrator.dirs["processed_data"], expected_output_filename)
        self.assertTrue(os.path.exists(output_file), f"Expected processed file {output_file} not found.")

        # Check state update
        self.assertTrue(self.orchestrator.state.get("preprocessing_completed"))
        self.assertEqual(self.orchestrator.state["molecules_processed"], len(TEST_SMILES))

        # Check checkpoint
        checkpoint_file = os.path.join(self.orchestrator.dirs["checkpoints"], "preprocessing_checkpoint.pkl")
        self.assertTrue(os.path.exists(checkpoint_file))

    def test_03_caching_mechanism(self):
        """Test caching mechanism."""
        logger.info("Testing caching mechanism...")

        # Clear orchestrator cache for the specific stage
        self.orchestrator.cache["processed_dataframe"] = None  # Updated cache key

        # Run preprocessing again without force_rerun
        start_time = time.time()
        df = self.orchestrator.run_preprocessing_stage(
            input_file=self.test_dataset, force_rerun=False
        )  # Updated method name
        end_time = time.time()

        # Verify results
        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(TEST_SMILES))

        # Verify it used cached results (should be very fast)
        self.assertLess(end_time - start_time, 0.5)  # Less than 0.5 seconds indicates cache was used

        # Force rerun and verify it takes longer
        start_time = time.time()
        df = self.orchestrator.run_preprocessing_stage(
            input_file=self.test_dataset, force_rerun=True
        )  # Updated method name
        end_time = time.time()

        # This should take longer as it's not using the cache, but be reasonable with timing threshold
        self.assertGreater(end_time - start_time, 0.001)  # Actual processing takes more time

    def test_04_orca_calculation_mocked(self):
        """Test ORCA calculation stage (mocked)."""
        logger.info("Testing ORCA calculation stage (mocked)...")

        # Enable mocking of ORCA calculations
        self.orchestrator.config["execution"]["skip_qm"] = True

        # Run ORCA calculations
        # The input_file for run_qm_stage should be the processed CSV from the preprocessing stage
        processed_csv_path = os.path.join(self.orchestrator.dirs["processed_data"], "molecules_processed.csv")
        # Ensure this file exists from a previous stage or create a dummy one for this test if isolated
        if not os.path.exists(processed_csv_path) and self.orchestrator.cache.get("processed_dataframe") is None:
            # Run preprocessing if not done, to ensure the input for QM stage is present
            logger.info("Preprocessing data for ORCA test setup as processed file not found.")
            self.orchestrator.run_preprocessing_stage(
                input_file=self.test_dataset, force_rerun=True
            )  # Use True if state is unknown

        # Now, the processed_csv_path should exist or its data be in cache
        orca_results = self.orchestrator.run_orca_calculations(input_file=processed_csv_path)  # Use input_file

        # Should return empty DataFrame when skipped
        self.assertIsInstance(orca_results, pd.DataFrame)
        self.assertEqual(len(orca_results), 0)

        # Disable mocking
        self.orchestrator.config["execution"]["skip_qm"] = False

    def test_05_full_pipeline_mocked(self):
        """Test full pipeline with mocked ORCA and graph generation."""
        logger.info("Testing full pipeline with mocked stages...")

        # Enable mocking of time-consuming stages
        self.orchestrator.config["execution"]["skip_qm"] = True
        self.orchestrator.config["execution"]["skip_graph_generation"] = True

        # Run full pipeline
        results = self.orchestrator.execute_pipeline(input_file=self.test_dataset, force_rerun=True)

        # Verify results
        self.assertIsNotNone(results)
        self.assertTrue("preprocessing" in results)
        self.assertEqual(results["preprocessing"]["total_compounds"], len(TEST_SMILES))
        self.assertEqual(results["preprocessing"]["valid_compounds"], 4)  # We have 4 valid SMILES in TEST_SMILES

        # Verify state is updated
        self.assertTrue(self.orchestrator.state.get("preprocessing_completed"))
        self.assertEqual(self.orchestrator.state["molecules_processed"], len(TEST_SMILES))

        # Disable mocking
        self.orchestrator.config["execution"]["skip_qm"] = False
        self.orchestrator.config["execution"]["skip_graph_generation"] = False

    def test_06_resume_pipeline(self):
        """Test resuming pipeline from a checkpoint."""
        logger.info("Testing pipeline resume functionality (for preprocessing stage)...")

        base_input_filename = os.path.splitext(os.path.basename(self.test_dataset))[0]
        expected_output_filename = f"{base_input_filename}_pfas_processed.csv"
        processed_csv_path = os.path.join(self.orchestrator.dirs["processed_data"], expected_output_filename)

        # Run preprocessing first with force_rerun to establish a baseline and save state
        self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=True)
        self.assertTrue(os.path.exists(processed_csv_path), f"Expected processed file {processed_csv_path} not found after initial run.")
        timestamp_run1 = os.path.getmtime(processed_csv_path)
        self.orchestrator.state.copy()

        # Allow a brief moment for timestamp granularity
        time.sleep(0.1)

        # Create a new orchestrator instance - it should load the previously saved state
        # (assuming state is saved to a file and loaded on init, which it is via _load_state)
        # For this test, we'll use the same orchestrator instance but simulate re-entry
        # by clearing a part of its internal cache that might make it re-run without state file.
        # More robustly, one would create a new instance.
        # For now, let's rely on force_rerun=False and the existing state file.

        # To ensure it's not just using in-memory cache from the same instance:
        self.orchestrator.cache["processed_dataframe"] = None

        # Run preprocessing again, force_rerun is False (default for run_preprocessing_stage if not specified)
        # The orchestrator should see that preprocessing is complete from its loaded state.
        df_run2 = self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=False)

        self.assertTrue(os.path.exists(processed_csv_path), f"Expected processed file {processed_csv_path} not found after resume run.")
        timestamp_run2 = os.path.getmtime(processed_csv_path)

        # Assert that the file was not modified, meaning the stage was skipped due to loaded state
        self.assertEqual(
            timestamp_run1, timestamp_run2, "Processed file was modified, stage did not resume from state."
        )

        # Verify results from the second run (should be loaded from cache/state)
        self.assertIsNotNone(df_run2)
        self.assertEqual(len(df_run2), len(TEST_SMILES))
        self.assertTrue(self.orchestrator.state.get("preprocessing_completed"))


def run_tests():
    """Run the test suite."""
    logger.info("Starting pipeline tests...")
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPFASPipelineOrchestrator)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    logger.info("Pipeline tests completed.")
    return len(result.errors) == 0 and len(result.failures) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
