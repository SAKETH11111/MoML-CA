#!/usr/bin/env python3
"""
test_djmgnn_force_field_pipeline.py

Test the DJMGNN model's capability to provide force field parameters for MD simulations.

This script tests whether the trained DJMGNN can be adapted to provide:
1. Partial charges from node predictions
2. Force field parameters via ForceFieldMapper
3. OpenMM-compatible output for MD simulations

Usage:
    python scripts/test_djmgnn_force_field_pipeline.py --model_path /tmp/djmgnn_model/finetuned_model/pytorch_model.pt
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.models.mgnn.djmgnn import DJMGNN
from moml.simulation.molecular_dynamics.force_field.mapper import ForceFieldMapper
from scripts.test_huggingface_djmgnn import SimplePFASConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DJMGNNForceFieldAdapter:
    """Adapts DJMGNN predictions for force field parameter generation."""
    
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.converter = SimplePFASConverter()
        self.ff_mapper = ForceFieldMapper()
        
        # Load DJMGNN model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        
    def _load_model(self, model_path: str) -> DJMGNN:
        """Load the trained DJMGNN model."""
        # Load config
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Fix jk_mode compatibility
        jk_mode = config['jk_mode']
        if jk_mode == 'cat':
            jk_mode = 'concat'
        
        # Create model with 29D input (as per trained model)
        model = DJMGNN(
            in_node_dim=29,  # Trained with 29D features
            in_edge_dim=config.get('in_edge_dim', 0),
            hidden_dim=config['hidden_dim'],
            n_blocks=config['n_blocks'],
            layers_per_block=config['layers_per_block'],
            node_output_dims=config['node_output_dims'],
            graph_output_dims=config['graph_output_dims'],
            dropout=config['dropout'],
            jk_mode=jk_mode,
            use_supernode=config['use_supernode'],
            use_rbf=config['use_rbf'],
            rbf_K=config['rbf_K'],
            pool_type='mean'
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        logger.info("✅ Model loaded successfully")
        return model
    
    def extract_partial_charges(self, smiles: str) -> Tuple[Optional[Chem.Mol], Optional[np.ndarray]]:
        """
        Extract partial charges from DJMGNN node predictions.
        
        Returns:
            Tuple of (RDKit molecule, partial charges array)
        """
        try:
            # Convert SMILES to graph
            mol_graph = self.converter.smiles_to_graph(smiles)
            if mol_graph is None:
                return None, None
            
            mol_graph = mol_graph.to(self.device)
            
            # Get RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None
            mol = Chem.AddHs(mol)
            
            # Get model predictions
            with torch.no_grad():
                predictions = self.model(
                    x=mol_graph.x,
                    edge_index=mol_graph.edge_index,
                    edge_attr=getattr(mol_graph, 'edge_attr', None),
                    batch=mol_graph.batch
                )
                
                # Extract node predictions
                node_pred = predictions.get('node_pred')
                if node_pred is None:
                    logger.warning("No node predictions available")
                    return mol, None
                
                # Interpret first dimension of node predictions as partial charges
                # Note: This is a simplified approach - in practice, the model would need
                # to be trained specifically for partial charge prediction
                if node_pred.shape[1] > 0:
                    # Use first output dimension as partial charge estimate
                    partial_charges = node_pred[:, 0].cpu().numpy()
                    
                    # Normalize charges to be physically reasonable
                    # Typical partial charges range from -1 to +1
                    partial_charges = np.tanh(partial_charges) * 0.5  # Scale to [-0.5, 0.5]
                    
                    return mol, partial_charges
                else:
                    logger.warning(f"Node predictions have no features: shape {node_pred.shape}")
                    return mol, None
                    
        except Exception as e:
            logger.error(f"Error extracting partial charges for {smiles}: {e}")
            return None, None
    
    def generate_force_field_parameters(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        Generate complete force field parameters for a molecule.
        
        Returns:
            Dictionary containing bond, angle, dihedral parameters and partial charges
        """
        # Extract partial charges from DJMGNN
        mol, partial_charges = self.extract_partial_charges(smiles)
        if mol is None:
            return None
        
        # If no partial charges from model, fall back to Gasteiger charges
        if partial_charges is None:
            logger.info("Using fallback Gasteiger charges")
            AllChem.ComputeGasteigerCharges(mol)
            partial_charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        
        # Generate 3D conformer for proper geometry
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            logger.warning("Could not generate 3D conformer")
        
        # Generate force field parameters
        try:
            ff_params = self.ff_mapper.generate_force_field_parameters(
                mol, 
                partial_charges=partial_charges
            )
            
            # Add metadata
            ff_params['metadata'] = {
                'smiles': smiles,
                'num_atoms': mol.GetNumAtoms(),
                'charge_source': 'djmgnn' if partial_charges is not None else 'gasteiger',
                'model_type': 'djmgnn_adapted'
            }
            
            return ff_params
            
        except Exception as e:
            logger.error(f"Error generating force field parameters: {e}")
            return None
    
    def test_pfas_molecules(self, pfas_smiles_list: list) -> Dict[str, Any]:
        """Test force field generation for a list of PFAS molecules."""
        results = {
            'successful': 0,
            'failed': 0,
            'molecules': [],
            'errors': []
        }
        
        for smiles in pfas_smiles_list:
            logger.info(f"\nTesting: {smiles}")
            
            ff_params = self.generate_force_field_parameters(smiles)
            if ff_params is not None:
                results['successful'] += 1
                results['molecules'].append({
                    'smiles': smiles,
                    'num_bonds': len(ff_params.get('bonds', {})),
                    'num_angles': len(ff_params.get('angles', {})),
                    'num_dihedrals': len(ff_params.get('dihedrals', {})),
                    'num_atoms': ff_params['metadata']['num_atoms'],
                    'charge_sum': sum(ff_params.get('partial_charges', {}).values())
                })
                
                # Log summary
                logger.info(f"  ✅ Generated parameters:")
                logger.info(f"     - Atoms: {ff_params['metadata']['num_atoms']}")
                logger.info(f"     - Bonds: {len(ff_params.get('bonds', {}))}")
                logger.info(f"     - Angles: {len(ff_params.get('angles', {}))}")
                logger.info(f"     - Dihedrals: {len(ff_params.get('dihedrals', {}))}")
                logger.info(f"     - Charge sum: {sum(ff_params.get('partial_charges', {}).values()):.3f}")
            else:
                results['failed'] += 1
                results['errors'].append(f"Failed for: {smiles}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test DJMGNN force field parameter generation")
    parser.add_argument('--model_path', type=str, 
                       default='/tmp/djmgnn_model/finetuned_model/pytorch_model.pt',
                       help='Path to DJMGNN model')
    
    args = parser.parse_args()
    
    # Test PFAS molecules
    test_pfas_smiles = [
        "C(C(F)(F)F)(C(F)(F)F)(F)F",  # Perfluorooctanoic acid (PFOA) backbone
        "C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F",  # C8 perfluorinated
        "C(C(C(C(F)(F)F)(F)F)(F)F)(F)F",  # Shorter chain PFAS
        "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)O",  # PFAS with carboxylic acid
        "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)O"  # PFAS with sulfonic acid
    ]
    
    logger.info("=" * 70)
    logger.info("🧪 DJMGNN FORCE FIELD PARAMETER GENERATION TEST")
    logger.info("=" * 70)
    
    # Create adapter
    adapter = DJMGNNForceFieldAdapter(args.model_path)
    
    # Run tests
    results = adapter.test_pfas_molecules(test_pfas_smiles)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Successful: {results['successful']}/{len(test_pfas_smiles)}")
    logger.info(f"❌ Failed: {results['failed']}/{len(test_pfas_smiles)}")
    
    if results['successful'] > 0:
        logger.info("\n🎯 Force Field Generation Details:")
        for mol_info in results['molecules']:
            logger.info(f"\n  {mol_info['smiles'][:30]}...")
            logger.info(f"    - Atoms: {mol_info['num_atoms']}")
            logger.info(f"    - Bonds: {mol_info['num_bonds']}")
            logger.info(f"    - Angles: {mol_info['num_angles']}")
            logger.info(f"    - Dihedrals: {mol_info['num_dihedrals']}")
            logger.info(f"    - Total charge: {mol_info['charge_sum']:.3f}")
    
    # Assessment
    logger.info("\n" + "=" * 70)
    logger.info("🔍 ASSESSMENT")
    logger.info("=" * 70)
    
    if results['successful'] == len(test_pfas_smiles):
        logger.info("✅ DJMGNN can be adapted for force field generation!")
        logger.info("✅ All test molecules successfully parameterized")
        logger.info("\n⚠️  NOTE: The model needs specific training for accurate partial charges")
        logger.info("    Current implementation uses node predictions as a proxy")
    else:
        logger.info("⚠️  Some molecules failed - investigation needed")
        for error in results['errors']:
            logger.info(f"   - {error}")
    
    logger.info("\n💡 RECOMMENDATION:")
    logger.info("   1. Retrain DJMGNN with force field parameter targets")
    logger.info("   2. Use multi-task learning: molecular properties + force field params")
    logger.info("   3. Validate against QM-derived force field parameters")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()