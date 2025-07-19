"""
tests/test_data_ingestion_and_processing.py

Tests PFAS data processing pipeline components, including SMILES validation,
dataset processing, descriptor calculation, ORCA parsing (mock), graph
generation (mock), and pipeline orchestration.
"""

import importlib.util
import logging
import os
import sys
import tempfile
import unittest
from typing import Any

import pandas as pd

# Set up the path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


def import_module(module_path: str, module_name: str) -> Any:
    """
    Import a module from a file path or module path.

    Args:
        module_path (str): Dotted path or file path to the module.
        module_name (str): Name to assign to the module.

    Returns:
        Any: The imported module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    try:
        if module_path.startswith("code."):
            try:
                module = importlib.import_module(module_path[5:])
                return module
            except ImportError:
                module = importlib.import_module(module_path)
                return module
        else:
            try:
                module = importlib.import_module(f"code.{module_path}")
                return module
            except ImportError:
                module = importlib.import_module(module_path)
                return module
    except ImportError as e:
        try:
            file_path = os.path.join(project_root, *module_path.split("."))
            if os.path.exists(f"{file_path}.py"):
                spec = importlib.util.spec_from_file_location(module_name, f"{file_path}.py")
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
            elif os.path.exists(file_path) and os.path.isdir(file_path):
                init_path = os.path.join(file_path, "__init__.py")
                if os.path.exists(init_path):
                    spec = importlib.util.spec_from_file_location(module_name, init_path)
                    if spec is not None and spec.loader is not None:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return module
        except Exception as e2:
            raise ImportError(f"Failed to import {module_path}: {e2}") from e
        raise ImportError(f"Failed to import {module_path}")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_data_processing")


class TestDataProcessing(unittest.TestCase):
    """
    Test cases for data processing components.
    """

    def setUp(self) -> None:
        """
        Set up test data and imports.
        """
        try:
            module = import_module(
                "utils.helper_functions.molecular.molecule_processing", "molecule_processing"
            )
            self.validate_smiles = getattr(module, "validate_smiles")
            self.process_dataset = getattr(module, "process_dataset")
            self.calculate_basic_descriptors = getattr(module, "calculate_basic_descriptors")
            logger.info("Successfully imported molecule_processing module")
        except ImportError:
            logger.error("Failed to import molecule_processing")
            self.skipTest("Could not import molecule_processing module")

        self.valid_smiles = "CC(F)(F)F"
        self.invalid_smiles = "invalid_smiles_string"

        self.sample_data = pd.DataFrame(
            {
                "common_name": ["Compound1", "Compound2", "Compound3"],
                "SMILES": ["CC(F)(F)F", "CC(Cl)CC", "invalid_smiles"],
            }
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_csv_path = os.path.join(self.temp_dir.name, "sample.csv")
        self.sample_data.to_csv(self.sample_csv_path, index=False)

    def tearDown(self) -> None:
        """
        Clean up after test.
        """
        self.temp_dir.cleanup()

    def test_validate_smiles(self) -> None:
        """
        Test the SMILES validation function.
        """
        logger.info("Testing SMILES validation...")
        is_valid, canonical, error = self.validate_smiles(self.valid_smiles)
        self.assertTrue(is_valid)
        self.assertIsNotNone(canonical)
        self.assertIsNone(error)
        logger.info(f"Valid SMILES test passed. Canonical SMILES: {canonical}")

        is_valid, canonical, error = self.validate_smiles(self.invalid_smiles)
        self.assertFalse(is_valid)
        self.assertIsNone(canonical)
        self.assertIsNotNone(error)
        logger.info(f"Invalid SMILES test passed. Error: {error}")

    def test_process_dataset(self) -> None:
        """
        Test dataset processing.
        """
        logger.info("Testing dataset processing...")
        processed_df = self.process_dataset(self.sample_csv_path)
        self.assertEqual(len(processed_df), 3)
        self.assertIn("is_valid_smiles", processed_df.columns)
        self.assertIn("canonical_smiles", processed_df.columns)
        self.assertIn("rdkit_mol", processed_df.columns)
        valid_count = processed_df["is_valid_smiles"].sum()
        self.assertEqual(valid_count, 2)
        logger.info(f"Dataset processing test passed. Found {valid_count} valid compounds.")

    def test_calculate_descriptors(self) -> None:
        """
        Test molecular descriptor calculation.
        """
        logger.info("Testing descriptor calculation...")
        processed_df = self.process_dataset(self.sample_csv_path)
        descriptors_df = self.calculate_basic_descriptors(processed_df)
        self.assertIn("molecular_weight", descriptors_df.columns)
        self.assertIn("logp", descriptors_df.columns)
        self.assertIn("num_heavy_atoms", descriptors_df.columns)
        self.assertIn("num_rotatable_bonds", descriptors_df.columns)
        valid_compounds = descriptors_df[descriptors_df["is_valid_smiles"]]
        self.assertTrue(valid_compounds["molecular_weight"].notna().all())
        logger.info("Descriptor calculation test passed.")

    def test_mock_orca_parsing(self) -> None:
        """
        Mock test for ORCA output parsing.
        """
        logger.info("Testing ORCA parsing (mock)...")
        try:
            module = import_module("utils.quantum.orca_parser", "orca_parser")
            self.assertIsNotNone(getattr(module, "parse_orca_output", None))
            logger.info("ORCA parser import successful.")
        except ImportError:
            logger.error("ORCA parser import failed")
            self.skipTest("Could not import orca_parser module")

    def test_mock_graph_generation(self) -> None:
        """
        Mock test for graph generation.
        """
        logger.info("Testing graph generation (mock)...")
        try:
            module = import_module("MGNN.utils.unified_graph_generator", "unified_graph_generator")
            self.assertIsNotNone(getattr(module, "mol_file_to_graph", None))
            self.assertIsNotNone(getattr(module, "create_graph_from_orca_data", None))
            logger.info("Graph generator import successful.")
        except ImportError:
            logger.error("Graph generator import failed")
            self.skipTest("Could not import unified_graph_generator module")

    def test_orchestrator_import(self) -> None:
        """
        Test that the pipeline orchestrator can be imported.
        """
        logger.info("Testing pipeline orchestrator import...")
        try:
            module = import_module("integration.orchestration.pfas_pipeline_orchestrator", "pfas_pipeline_orchestrator")
            self.assertIsNotNone(getattr(module, "PFASPipelineOrchestrator", None))
            orchestrator_class = getattr(module, "PFASPipelineOrchestrator")
            orchestrator_class()
            logger.info("Pipeline orchestrator import and instantiation successful.")
        except ImportError:
            logger.error("Pipeline orchestrator import failed")
            self.skipTest("Could not import pfas_pipeline_orchestrator module")


def run_data_processing_pipeline_tests() -> bool:
    """
    Test the entire data processing pipeline with sample data.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    logger.info("\n========== RUNNING DATA PROCESSING PIPELINE TEST ==========")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataProcessing)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    logger.info("\n========== TEST RESULTS ==========")
    logger.info(f"Ran {result.testsRun} tests")
    if result.wasSuccessful():
        logger.info("All tests PASSED!")
        return True
    else:
        logger.error(f"Tests FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        for failure in result.failures:
            logger.error(f"FAILURE: {failure[0]} - {failure[1]}")
        for error in result.errors:
            logger.error(f"ERROR: {error[0]} - {error[1]}")
        return False


if __name__ == "__main__":
    success = run_data_processing_pipeline_tests()
    sys.exit(0 if success else 1)
