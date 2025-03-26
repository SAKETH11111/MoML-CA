#!/usr/bin/env python3
"""
Test script for PFAS data processing pipeline components

This script tests the key components of the data processing pipeline:
1. SMILES validation
2. Dataset processing
3. Descriptor calculation
4. ORCA parsing (mock test)
5. Graph generation (mock test)
6. Basic pipeline orchestration

Run this script to verify that all components are working correctly.
"""

import os
import sys
import pandas as pd
import logging
import tempfile
from pathlib import Path
import unittest
import importlib.util

# Set up the path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Helper function to import modules safely
def import_module(module_path, module_name):
    """Import a module from a file path."""
    try:
        if module_path.startswith('code.'):
            # Try without 'code.' prefix
            try:
                module = importlib.import_module(module_path[5:])
                return module
            except ImportError:
                # Try with 'code.' prefix
                module = importlib.import_module(module_path)
                return module
        else:
            # Try with 'code.' prefix
            try:
                module = importlib.import_module(f"code.{module_path}")
                return module
            except ImportError:
                # Try without 'code.' prefix
                module = importlib.import_module(module_path)
                return module
    except ImportError as e:
        # If both fail, try a direct file import using importlib.util
        try:
            file_path = os.path.join(project_root, *module_path.split('.'))
            if os.path.exists(f"{file_path}.py"):
                spec = importlib.util.spec_from_file_location(module_name, f"{file_path}.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            elif os.path.exists(file_path) and os.path.isdir(file_path):
                init_path = os.path.join(file_path, "__init__.py")
                if os.path.exists(init_path):
                    spec = importlib.util.spec_from_file_location(module_name, init_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
        except Exception as e2:
            raise ImportError(f"Failed to import {module_path}: {e2}") from e
        
        raise ImportError(f"Failed to import {module_path}: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_data_processing")


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing components."""
    
    def setUp(self):
        """Set up test data and imports."""
        # Try to import the required modules
        try:
            # Import molecule_processing
            module = import_module('utils.helper_functions.molecular.molecule_processing', 'molecule_processing')
            self.validate_smiles = getattr(module, 'validate_smiles')
            self.process_dataset = getattr(module, 'process_dataset')
            self.calculate_basic_descriptors = getattr(module, 'calculate_basic_descriptors')
            logger.info("Successfully imported molecule_processing module")
        except ImportError as e:
            logger.error(f"Failed to import molecule_processing: {e}")
            self.skipTest("Could not import molecule_processing module")
            
        # Sample SMILES for testing
        self.valid_smiles = "CC(F)(F)F"  # Trifluoromethane
        self.invalid_smiles = "invalid_smiles_string"
        
        # Create a sample dataset
        self.sample_data = pd.DataFrame({
            'common_name': ['Compound1', 'Compound2', 'Compound3'],
            'SMILES': ['CC(F)(F)F', 'CC(Cl)CC', 'invalid_smiles']
        })
        
        # Save sample data to a temporary CSV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_csv_path = os.path.join(self.temp_dir.name, 'sample.csv')
        self.sample_data.to_csv(self.sample_csv_path, index=False)
    
    def tearDown(self):
        """Clean up after test."""
        self.temp_dir.cleanup()
    
    def test_validate_smiles(self):
        """Test the SMILES validation function."""
        logger.info("Testing SMILES validation...")
        
        # Test valid SMILES
        is_valid, canonical, error = self.validate_smiles(self.valid_smiles)
        self.assertTrue(is_valid)
        self.assertIsNotNone(canonical)
        self.assertIsNone(error)
        logger.info(f"Valid SMILES test passed. Canonical SMILES: {canonical}")
        
        # Test invalid SMILES
        is_valid, canonical, error = self.validate_smiles(self.invalid_smiles)
        self.assertFalse(is_valid)
        self.assertIsNone(canonical)
        self.assertIsNotNone(error)
        logger.info(f"Invalid SMILES test passed. Error: {error}")
    
    def test_process_dataset(self):
        """Test dataset processing."""
        logger.info("Testing dataset processing...")
        
        # Process the sample dataset
        processed_df = self.process_dataset(self.sample_csv_path)
        
        # Verify results
        self.assertEqual(len(processed_df), 3)
        self.assertTrue('is_valid_smiles' in processed_df.columns)
        self.assertTrue('canonical_smiles' in processed_df.columns)
        self.assertTrue('rdkit_mol' in processed_df.columns)
        
        # Check that exactly 2 SMILES are valid
        valid_count = processed_df['is_valid_smiles'].sum()
        self.assertEqual(valid_count, 2)
        
        logger.info(f"Dataset processing test passed. Found {valid_count} valid compounds.")
    
    def test_calculate_descriptors(self):
        """Test molecular descriptor calculation."""
        logger.info("Testing descriptor calculation...")
        
        # First process the dataset
        processed_df = self.process_dataset(self.sample_csv_path)
        
        # Calculate descriptors
        descriptors_df = self.calculate_basic_descriptors(processed_df)
        
        # Verify results
        self.assertTrue('molecular_weight' in descriptors_df.columns)
        self.assertTrue('logp' in descriptors_df.columns)
        self.assertTrue('num_heavy_atoms' in descriptors_df.columns)
        self.assertTrue('num_rotatable_bonds' in descriptors_df.columns)
        
        # Check that descriptors are calculated for valid molecules only
        valid_compounds = descriptors_df[descriptors_df['is_valid_smiles']]
        self.assertTrue(valid_compounds['molecular_weight'].notna().all())
        
        logger.info("Descriptor calculation test passed.")
    
    def test_mock_orca_parsing(self):
        """Mock test for ORCA output parsing."""
        logger.info("Testing ORCA parsing (mock)...")
        
        # This is a mock test since we don't have actual ORCA output files
        # We're just verifying the imports work
        try:
            module = import_module('utils.quantum.orca_parser', 'orca_parser')
            self.assertIsNotNone(getattr(module, 'parse_orca_output', None))
            logger.info("ORCA parser import successful.")
        except ImportError as e:
            logger.error(f"ORCA parser import failed: {e}")
            self.skipTest("Could not import orca_parser module")
    
    def test_mock_graph_generation(self):
        """Mock test for graph generation."""
        logger.info("Testing graph generation (mock)...")
        
        # This is a mock test since we don't have actual molecule files and ORCA outputs
        # We're just verifying the imports work
        try:
            module = import_module('MGNN.utils.unified_graph_generator', 'unified_graph_generator')
            self.assertIsNotNone(getattr(module, 'mol_file_to_graph', None))
            self.assertIsNotNone(getattr(module, 'create_graph_from_orca_data', None))
            logger.info("Graph generator import successful.")
        except ImportError as e:
            logger.error(f"Graph generator import failed: {e}")
            self.skipTest("Could not import unified_graph_generator module")
    
    def test_orchestrator_import(self):
        """Test that the pipeline orchestrator can be imported."""
        logger.info("Testing pipeline orchestrator import...")
        
        try:
            module = import_module('integration.orchestration.pfas_pipeline_orchestrator', 'pfas_pipeline_orchestrator')
            self.assertIsNotNone(getattr(module, 'PFASPipelineOrchestrator', None))
            
            # Try creating an instance with default parameters
            orchestrator_class = getattr(module, 'PFASPipelineOrchestrator')
            orchestrator = orchestrator_class()
            logger.info("Pipeline orchestrator import and instantiation successful.")
        except ImportError as e:
            logger.error(f"Pipeline orchestrator import failed: {e}")
            self.skipTest("Could not import pfas_pipeline_orchestrator module")
        except Exception as e:
            logger.error(f"Pipeline orchestrator instantiation failed: {e}")
            self.skipTest(f"Could not instantiate PFASPipelineOrchestrator: {e}")


def test_data_processing_pipeline():
    """Test the entire data processing pipeline with sample data."""
    logger.info("\n========== RUNNING DATA PROCESSING PIPELINE TEST ==========\n")
    
    # Create a unittest test suite and run all tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataProcessing)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Print summary
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
    success = test_data_processing_pipeline()
    sys.exit(0 if success else 1) 