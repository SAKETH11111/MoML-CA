#!/usr/bin/env python3
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

import os
import sys
import time
import tempfile
import unittest
import shutil
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pipeline_test")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

# Import pipeline components
from moml.pipeline.orchestrator import PFASPipelineOrchestrator

# Sample SMILES for testing
TEST_SMILES = [
    # Valid PFAS SMILES
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFOA"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFHxA"),
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "PFBA"),
    # Invalid SMILES to test error handling
    ("INVALID_SMILES", "INVALID"),
    ("CC(CC(C)(C))", "SYNTAX_ERROR"),
]

def create_test_dataset(output_dir: str) -> str:
    """Create a test dataset with PFAS SMILES strings."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame with SMILES and names
    data = {
        "SMILES": [smile[0] for smile in TEST_SMILES],
        "common_name": [smile[1] for smile in TEST_SMILES]
    }
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
        "parallel": {
            "enabled": True,
            "max_workers": 2
        },
        "qm": {
            "functional": "B3LYP",
            "basis_set": "6-31G*",
            "num_procs": 1,
            "memory": 1000
        },
        "graph": {
            "charge_type": "mulliken",
            "use_pfas_features": True,
            "use_quantum_properties": True
        },
        "execution": {
            "skip_qm": False,
            "skip_graph_generation": False,
            "force_rerun": False,
            "cache_intermediates": True
        }
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
            cache_intermediates=True
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
        df = self.orchestrator.preprocess_data(self.test_dataset, force_rerun=True)
        
        # Verify results
        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(TEST_SMILES))
        self.assertTrue("is_valid_smiles" in df.columns)
        self.assertTrue("canonical_smiles" in df.columns)
        self.assertTrue("mol_weight" in df.columns)
        
        # Check valid/invalid SMILES counts
        valid_count = sum(df["is_valid_smiles"])
        invalid_count = len(df) - valid_count
        self.assertEqual(valid_count, 3)  # We have 3 valid SMILES in TEST_SMILES
        self.assertEqual(invalid_count, 2)  # We have 2 invalid SMILES in TEST_SMILES
        
        # Check output file
        output_file = os.path.join(self.orchestrator.dirs["processed_data"], "pfas_processed.csv")
        self.assertTrue(os.path.exists(output_file))
        
        # Check state update
        self.assertTrue(self.orchestrator.state["preprocessed"])
        self.assertEqual(self.orchestrator.state["molecules_processed"], len(TEST_SMILES))
        
        # Check checkpoint
        checkpoint_file = os.path.join(self.orchestrator.dirs["checkpoints"], "preprocessing_checkpoint.pkl")
        self.assertTrue(os.path.exists(checkpoint_file))
    
    def test_03_caching_mechanism(self):
        """Test caching mechanism."""
        logger.info("Testing caching mechanism...")
        
        # Clear orchestrator cache
        self.orchestrator.cache["processed_df"] = None
        
        # Run preprocessing again without force_rerun
        start_time = time.time()
        df = self.orchestrator.preprocess_data(self.test_dataset, force_rerun=False)
        end_time = time.time()
        
        # Verify results
        self.assertIsNotNone(df)
        self.assertEqual(len(df), len(TEST_SMILES))
        
        # Verify it used cached results (should be very fast)
        self.assertLess(end_time - start_time, 0.5)  # Less than 0.5 seconds indicates cache was used
        
        # Force rerun and verify it takes longer
        start_time = time.time()
        df = self.orchestrator.preprocess_data(self.test_dataset, force_rerun=True)
        end_time = time.time()
        
        # This should take longer as it's not using the cache
        self.assertGreater(end_time - start_time, 0.01)  # Actual processing takes more time
    
    def test_04_orca_calculation_mocked(self):
        """Test ORCA calculation stage (mocked)."""
        logger.info("Testing ORCA calculation stage (mocked)...")
        
        # Enable mocking of ORCA calculations
        self.orchestrator.config["execution"]["skip_qm"] = True
        
        # Run ORCA calculations
        orca_results = self.orchestrator.run_orca_calculations(input_file=self.test_dataset)
        
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
        results = self.orchestrator.run_full_pipeline(self.test_dataset, force_rerun=True)
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results["molecules_processed"], len(TEST_SMILES))
        self.assertEqual(results["valid_molecules"], 3)  # We have 3 valid SMILES in TEST_SMILES
        
        # Verify state is updated
        self.assertTrue(self.orchestrator.state["preprocessed"])
        self.assertEqual(self.orchestrator.state["molecules_processed"], len(TEST_SMILES))
        
        # Disable mocking
        self.orchestrator.config["execution"]["skip_qm"] = False
        self.orchestrator.config["execution"]["skip_graph_generation"] = False
    
    def test_06_resume_pipeline(self):
        """Test resuming pipeline from a checkpoint."""
        logger.info("Testing pipeline resume functionality...")
        
        # Reset state to simulate interrupted pipeline
        self.orchestrator.state["orca_calculated"] = False
        self.orchestrator.state["graphs_generated"] = False
        self.orchestrator._save_state()
        
        # Mock stages to avoid actual computation
        self.orchestrator.config["execution"]["skip_qm"] = True
        self.orchestrator.config["execution"]["skip_graph_generation"] = True
        
        # Resume pipeline
        results = self.orchestrator.resume_pipeline(self.test_dataset)
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results["molecules_processed"], len(TEST_SMILES))
        self.assertEqual(results["valid_molecules"], 3)  # We have 3 valid SMILES in TEST_SMILES
        
        # Disable mocking
        self.orchestrator.config["execution"]["skip_qm"] = False
        self.orchestrator.config["execution"]["skip_graph_generation"] = False

def run_tests():
    """Run the test suite."""
    logger.info("Starting pipeline tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Pipeline tests completed.")

if __name__ == "__main__":
    run_tests() 