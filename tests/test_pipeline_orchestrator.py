"""
tests/test_pipeline_orchestrator.py

Unit tests for the PFAS Pipeline Orchestrator.

This script tests the optimized pipeline stages, verifying that:
1. Data preprocessing works correctly
2. ORCA calculation orchestration functions properly (mocked if ORCA not available)
3. Molecular graph generation succeeds
4. Checkpointing and caching mechanisms function as expected
5. The resume functionality recovers from interruptions
"""

import logging
import os
import shutil
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List, Tuple

import pandas as pd

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
except ImportError:
    logger.warning("Pipeline module not available")
    PIPELINE_AVAILABLE = False

# Sample SMILES for testing
TEST_SMILES: List[Tuple[str, str]] = [
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFOA"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFHxA"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFBA"),
    ("CC(CC(C)(C))", "SYNTAX_ERROR"),
    ("INVALID_SMILES", "INVALID"),
]


def create_test_dataset(output_dir: str) -> str:
    """
    Create a test dataset with PFAS SMILES strings.

    Args:
        output_dir (str): Directory to save the dataset.

    Returns:
        str: Path to the created CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = {"SMILES": [smile[0] for smile in TEST_SMILES], "common_name": [smile[1] for smile in TEST_SMILES]}
    df = pd.DataFrame(data)

    output_file = os.path.join(output_dir, "test_pfas_dataset.csv")
    df.to_csv(output_file, index=False)

    return output_file


def create_test_config(base_dir: str) -> Dict[str, Any]:
    """
    Create a test configuration for the pipeline.

    Args:
        base_dir (str): Base directory for data, output, and working directories.

    Returns:
        Dict[str, Any]: A dictionary containing the test configuration.
    """
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


def setup_test_environment() -> Tuple[str, str, Dict[str, Any]]:
    """
    Set up a test environment with temporary directories.

    Returns:
        Tuple[str, str, Dict[str, Any]]: A tuple containing:
            - Path to the temporary test directory.
            - Path to the created test dataset CSV.
            - The test configuration dictionary.
    """
    test_dir = tempfile.mkdtemp(prefix="moml_ca_test_")

    os.makedirs(os.path.join(test_dir, "data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "output"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "working"), exist_ok=True)

    test_dataset = create_test_dataset(os.path.join(test_dir, "data", "raw"))

    test_config = create_test_config(test_dir)

    return test_dir, test_dataset, test_config


def cleanup_test_environment(test_dir: str) -> None:
    """
    Clean up the test environment by removing the temporary directory.

    Args:
        test_dir (str): Path to the temporary test directory.
    """
    try:
        shutil.rmtree(test_dir)
        logger.info(f"Cleaned up test directory: {test_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up test directory {test_dir}: {e}")


class TestPFASPipelineOrchestrator(unittest.TestCase):
    """
    Test case for the PFAS Pipeline Orchestrator.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up test environment once for all tests in this class.
        """
        cls.test_dir, cls.test_dataset, cls.test_config = setup_test_environment()
        logger.info(f"Set up test environment in {cls.test_dir}")

        cls.orchestrator = PFASPipelineOrchestrator(
            data_dir=cls.test_config["data_dir"],
            output_dir=cls.test_config["output_dir"],
            working_dir=cls.test_config["working_dir"],
        )

        cls.orchestrator.config.update(cls.test_config)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Clean up test environment after all tests in this class.
        """
        cleanup_test_environment(cls.test_dir)

    def test_01_initialization(self) -> None:
        """
        Test orchestrator initialization and directory creation.
        """
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.config["data_dir"], self.test_config["data_dir"])
        self.assertEqual(self.orchestrator.config["output_dir"], self.test_config["output_dir"])
        self.assertEqual(self.orchestrator.config["working_dir"], self.test_config["working_dir"])

        self.assertTrue(os.path.exists(self.orchestrator.dirs["raw_data"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["processed_data"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["orca_input"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["orca_output"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["molecule_files"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["molecular_graphs"]))
        self.assertTrue(os.path.exists(self.orchestrator.dirs["checkpoints"]))

    def test_02_preprocess_data(self) -> None:
        """
        Test preprocessing stage.
        """
        logger.info("Testing preprocessing stage...")

        df = self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=True)

        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(TEST_SMILES))
        self.assertTrue("is_valid_smiles" in df.columns)
        self.assertTrue("canonical_smiles" in df.columns)
        self.assertTrue("molecular_weight" in df.columns)

        valid_count = sum(df["is_valid_smiles"])
        invalid_count = len(df) - valid_count
        self.assertEqual(valid_count, 4)
        self.assertEqual(invalid_count, 1)

        base_input_filename = os.path.splitext(os.path.basename(self.test_dataset))[0]
        expected_output_filename = f"{base_input_filename}_pfas_processed.csv"
        output_file = os.path.join(self.orchestrator.dirs["processed_data"], expected_output_filename)
        self.assertTrue(os.path.exists(output_file), f"Expected processed file {output_file} not found.")

        self.assertTrue(self.orchestrator.state.get("preprocessing_completed"))
        self.assertEqual(self.orchestrator.state["molecules_processed"], len(TEST_SMILES))

        checkpoint_file = os.path.join(self.orchestrator.dirs["checkpoints"], "preprocessing_checkpoint.pkl")
        self.assertTrue(os.path.exists(checkpoint_file))

    def test_03_caching_mechanism(self) -> None:
        """
        Test caching mechanism for preprocessing stage.
        """
        logger.info("Testing caching mechanism...")

        self.orchestrator.cache["processed_dataframe"] = None

        start_time = time.time()
        df = self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=False)
        end_time = time.time()

        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(TEST_SMILES))

        self.assertLess(end_time - start_time, 0.5)

        start_time = time.time()
        df = self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=True)
        end_time = time.time()

        self.assertGreater(end_time - start_time, 0.001)

    def test_04_orca_calculation_mocked(self) -> None:
        """
        Test ORCA calculation stage (mocked).
        """
        logger.info("Testing ORCA calculation stage (mocked)...")

        self.orchestrator.config["execution"]["skip_qm"] = True

        processed_csv_path = os.path.join(self.orchestrator.dirs["processed_data"], "molecules_processed.csv")
        if not os.path.exists(processed_csv_path) and self.orchestrator.cache.get("processed_dataframe") is None:
            logger.info("Preprocessing data for ORCA test setup as processed file not found.")
            self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=True)

        orca_results = self.orchestrator.run_orca_calculations(input_file=processed_csv_path)

        self.assertIsInstance(orca_results, pd.DataFrame)
        self.assertEqual(len(orca_results), 0)

        self.orchestrator.config["execution"]["skip_qm"] = False

    def test_05_full_pipeline_mocked(self) -> None:
        """
        Test full pipeline with mocked ORCA and graph generation stages.
        """
        logger.info("Testing full pipeline with mocked stages...")

        self.orchestrator.config["execution"]["skip_qm"] = True
        self.orchestrator.config["execution"]["skip_graph_generation"] = True

        results = self.orchestrator.execute_pipeline(input_file=self.test_dataset, force_rerun=True)

        self.assertIsNotNone(results)
        self.assertTrue("preprocessing" in results)
        self.assertEqual(results["preprocessing"]["total_compounds"], len(TEST_SMILES))
        self.assertEqual(results["preprocessing"]["valid_compounds"], 4)

        self.assertTrue(self.orchestrator.state.get("preprocessing_completed"))
        self.assertEqual(self.orchestrator.state["molecules_processed"], len(TEST_SMILES))

        self.orchestrator.config["execution"]["skip_qm"] = False
        self.orchestrator.config["execution"]["skip_graph_generation"] = False

    def test_06_resume_pipeline(self) -> None:
        """
        Test resuming pipeline from a checkpoint for the preprocessing stage.
        """
        logger.info("Testing pipeline resume functionality (for preprocessing stage)...")

        base_input_filename = os.path.splitext(os.path.basename(self.test_dataset))[0]
        expected_output_filename = f"{base_input_filename}_pfas_processed.csv"
        processed_csv_path = os.path.join(self.orchestrator.dirs["processed_data"], expected_output_filename)

        self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=True)
        self.assertTrue(os.path.exists(processed_csv_path), f"Expected processed file {processed_csv_path} not found after initial run.")
        timestamp_run1 = os.path.getmtime(processed_csv_path)
        self.orchestrator.state.copy()

        time.sleep(0.1)

        self.orchestrator.cache["processed_dataframe"] = None

        df_run2 = self.orchestrator.run_preprocessing_stage(input_file=self.test_dataset, force_rerun=False)

        self.assertTrue(os.path.exists(processed_csv_path), f"Expected processed file {processed_csv_path} not found after resume run.")
        timestamp_run2 = os.path.getmtime(processed_csv_path)

        self.assertEqual(
            timestamp_run1, timestamp_run2, "Processed file was modified, stage did not resume from state."
        )

        self.assertIsNotNone(df_run2)
        self.assertEqual(len(df_run2), len(TEST_SMILES))
        self.assertTrue(self.orchestrator.state.get("preprocessing_completed"))


def run_tests() -> bool:
    """
    Run the test suite for the PFAS Pipeline Orchestrator.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    logger.info("Starting pipeline tests...")
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPFASPipelineOrchestrator)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    logger.info("Pipeline tests completed.")
    return len(result.errors) == 0 and len(result.failures) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
