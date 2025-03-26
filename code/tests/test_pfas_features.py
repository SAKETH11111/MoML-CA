#!/usr/bin/env python3
"""
Test script for PFAS-specific features and visualization

This script tests:
1. Creating graphs with PFAS-specific features
2. Visualizing molecular graphs with different highlighting options
3. Calculating and verifying PFAS-specific properties
"""

import os
import sys
import unittest
import tempfile
import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_pfas_features")

# Check for RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    print("RDKit import successful!")
except ImportError:
    print("Failed to import RDKit. Please make sure it's installed.")
    sys.exit(1)

# Try to import torch and matplotlib (for visualization)
try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch import successful!")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"Failed to import PyTorch: {e}")

# Try to import matplotlib for visualization testing
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib import successful!")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"Failed to import Matplotlib: {e}")


def create_pfas_graph(smiles, include_visualization=False):
    """Create a graph representation with PFAS-specific features."""
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Count fluorine atoms
    f_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'F']
    f_count = len(f_atoms)
    
    # Calculate distance to fluorine atoms for each atom
    f_distance_features = {}
    if f_count > 0:
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            pos = conf.GetAtomPosition(atom_idx)
            
            # Calculate distances to all F atoms
            f_distances = []
            for f_idx in f_atoms:
                if f_idx == atom_idx:
                    continue
                f_pos = conf.GetAtomPosition(f_idx)
                dx = pos.x - f_pos.x
                dy = pos.y - f_pos.y
                dz = pos.z - f_pos.z
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                f_distances.append(dist)
            
            # Features: min distance, avg distance, close F count
            if f_distances:
                min_dist = min(f_distances)
                avg_dist = sum(f_distances) / len(f_distances)
                close_f = sum(1 for d in f_distances if d < 3.0)
            else:
                min_dist = 0.0
                avg_dist = 0.0
                close_f = 0
                
            f_distance_features[atom_idx] = [min_dist, avg_dist, float(close_f)]
    
    # Create a simple dictionary-based graph representation
    graph = {
        'num_nodes': mol.GetNumAtoms(),
        'f_count': f_count,
        'f_distance_features': f_distance_features,
        'smiles': smiles,
        'mol': mol
    }
    
    # Create visualization if requested and matplotlib is available
    if include_visualization and MATPLOTLIB_AVAILABLE:
        try:
            # Generate a simple 2D depiction
            AllChem.Compute2DCoords(mol)
            
            # Prepare the visualization
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Generate atom coordinates
            coords = {}
            for atom_idx in range(mol.GetNumAtoms()):
                pos = mol.GetConformer().GetAtomPosition(atom_idx)
                coords[atom_idx] = (pos.x, pos.y)
            
            # Draw bonds
            for bond in mol.GetBonds():
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                x1, y1 = coords[idx1]
                x2, y2 = coords[idx2]
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.0)
            
            # Draw atoms with different colors for fluorine
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                x, y = coords[atom_idx]
                
                if atom.GetSymbol() == 'F':
                    color = 'green'
                    size = 300
                elif atom.GetSymbol() == 'C':
                    # Check if this C is bonded to F
                    if any(mol.GetAtomWithIdx(b.GetOtherAtomIdx(atom_idx)).GetSymbol() == 'F' 
                           for b in atom.GetBonds()):
                        color = 'orange'
                        size = 250
                    else:
                        color = 'grey'
                        size = 200
                else:
                    color = 'blue'
                    size = 200
                
                ax.scatter(x, y, color=color, s=size, zorder=2)
                ax.text(x, y, atom.GetSymbol(), ha='center', va='center', fontsize=10, zorder=3)
            
            ax.axis('off')
            ax.set_title(f'PFAS Molecule: {f_count} F atoms')
            
            graph['visualization'] = fig
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    return graph


class TestPFASFeatures(unittest.TestCase):
    """Test PFAS-specific features and visualization."""
    
    def setUp(self):
        """Set up test molecules."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a few example PFAS SMILES strings
        self.test_smiles = [
            "CC(F)(F)F",  # Trifluoromethane (simple PFAS)
            "C(C(F)(F)F)C(F)(F)F",  # Hexafluoroethane
            "O=C(O)C(F)(OC(F)(F)C(F)(F)F)C(F)(F)F",  # GenX (complex PFAS)
            "CC",  # Ethane (non-PFAS control)
        ]
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
        
        # Close any open matplotlib figures
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')
    
    def test_pfas_detection(self):
        """Test detection of PFAS compounds based on fluorine count."""
        for idx, smiles in enumerate(self.test_smiles):
            graph = create_pfas_graph(smiles)
            
            # Check that we got a valid graph
            self.assertIsNotNone(graph)
            
            # Check fluorine counting
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            expected_f_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])
            self.assertEqual(graph['f_count'], expected_f_count)
            
            # Determine if it's a PFAS (has fluorine atoms)
            is_pfas = expected_f_count > 0
            pfas_status = "PFAS" if is_pfas else "non-PFAS"
            
            print(f"Molecule {idx+1}: {pfas_status} with {expected_f_count} fluorine atoms")
    
    def test_distance_features(self):
        """Test PFAS-specific distance features."""
        # Use a PFAS molecule
        pfas_idx = 2  # GenX (complex PFAS)
        graph = create_pfas_graph(self.test_smiles[pfas_idx])
        
        # Check distance features
        self.assertIn('f_distance_features', graph)
        
        # For PFAS, some atoms should have non-zero distance features
        non_zero_features = 0
        for atom_idx, features in graph['f_distance_features'].items():
            if features[0] > 0 or features[1] > 0 or features[2] > 0:
                non_zero_features += 1
        
        # At least some atoms should have non-zero distance features
        self.assertGreater(non_zero_features, 0, 
                         f"No non-zero distance features found in PFAS molecule (f_count={graph['f_count']})")
        
        print(f"Found {non_zero_features} atoms with non-zero fluorine distance features")
        
        # For non-PFAS, all distance features should be zeros
        non_pfas_idx = 3  # Ethane
        non_pfas_graph = create_pfas_graph(self.test_smiles[non_pfas_idx])
        
        if non_pfas_graph['f_count'] == 0:  # Confirm it's non-PFAS
            for atom_idx, features in non_pfas_graph['f_distance_features'].items():
                self.assertEqual(features[0], 0.0, "Non-PFAS should have zero min distance to F")
                self.assertEqual(features[1], 0.0, "Non-PFAS should have zero avg distance to F")
                self.assertEqual(features[2], 0.0, "Non-PFAS should have zero close F count")
    
    def test_visualization(self):
        """Test visualization of PFAS molecules."""
        if not MATPLOTLIB_AVAILABLE:
            self.skipTest("Matplotlib not available")
        
        for idx, smiles in enumerate(self.test_smiles[:-1]):  # Skip non-PFAS for visualization
            # Create graph with visualization
            graph = create_pfas_graph(smiles, include_visualization=True)
            
            # Check that visualization was created
            self.assertIn('visualization', graph, f"No visualization created for molecule {idx+1}")
            
            # Save the visualization to a file
            if 'visualization' in graph:
                output_path = os.path.join(self.temp_dir.name, f"molecule_{idx+1}.png")
                graph['visualization'].savefig(output_path)
                
                # Check file exists and has non-zero size
                self.assertTrue(os.path.exists(output_path), f"Visualization file not created: {output_path}")
                self.assertGreater(os.path.getsize(output_path), 0, f"Visualization file is empty: {output_path}")
                
                print(f"Visualization created for molecule {idx+1}: {output_path}")


def run_pfas_feature_tests():
    """Run the test suite."""
    # Create a test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPFASFeatures))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()


if __name__ == "__main__":
    print("\nTesting PFAS-specific features and visualization...")
    success = run_pfas_feature_tests()
    
    if success:
        print("\nAll PFAS feature tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome PFAS feature tests FAILED!")
        sys.exit(1) 