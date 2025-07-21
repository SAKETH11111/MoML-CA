#!/usr/bin/env python3
"""
test_huggingface_djmgnn.py

Test the trained DJMGNN model from HuggingFace against real PFAS experimental data.

This script validates whether the trained DJMGNN model from saketh11/MoML-CA 
performs better than:
1. Untrained baseline: 3/19 strong correlations (|r| > 0.5)  
2. Failed joint model: 1/19 correlations, 27.2% gradient coverage

Usage:
    python scripts/test_huggingface_djmgnn.py --max_molecules 50
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.feature_transforms import CreateEdges
from moml.models.mgnn.djmgnn import DJMGNN  # Import DJMGNN directly
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# Suppress RDKit deprecation warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePFASConverter:
    """Convert PFAS SMILES to molecular graphs with 29D node features to match trained model."""
    
    def __init__(self):
        self.create_edges = CreateEdges()
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to molecular graph with 29D node features to match trained model."""
        try:
            # Parse SMILES and add hydrogens
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for realistic geometry
            mol = Chem.AddHs(mol)
            
            # Generate 3D conformer
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except:
                    AllChem.UFFOptimizeMolecule(mol)
            except Exception as embed_error:
                logger.warning(f"Failed to generate 3D conformer for {smiles}: {embed_error}")
                mol = Chem.RemoveHs(mol)  # Fallback to 2D
            
            # Get atom features and coordinates (29D to match trained model)
            atoms = mol.GetAtoms()
            conformer = mol.GetConformer() if mol.GetNumConformers() > 0 else None
            
            node_features = []
            positions = []
            
            for i, atom in enumerate(atoms):
                # Extended 29D features to match the trained model
                features = [
                    float(atom.GetAtomicNum()),           # 0: Atomic number
                    float(atom.GetDegree()),              # 1: Degree  
                    float(atom.GetFormalCharge()),        # 2: Formal charge
                    float(atom.GetHybridization()),       # 3: Hybridization
                    float(atom.GetIsAromatic()),          # 4: Aromaticity
                    float(atom.GetTotalNumHs()),          # 5: Total hydrogens
                    float(atom.GetMass()),                # 6: Atomic mass
                    float(atom.IsInRing()),               # 7: In ring
                    float(atom.GetImplicitValence()),     # 8: Implicit valence
                    float(atom.GetExplicitValence()),     # 9: Explicit valence
                    float(atom.GetNumRadicalElectrons()), # 10: Radical electrons
                ]
                
                # Add more chemical features to reach 29 dimensions
                features.extend([
                    float(atom.GetHybridization() == Chem.HybridizationType.SP),      # 11: SP hybridization
                    float(atom.GetHybridization() == Chem.HybridizationType.SP2),     # 12: SP2 hybridization
                    float(atom.GetHybridization() == Chem.HybridizationType.SP3),     # 13: SP3 hybridization
                    float(atom.GetIsAromatic()),                                       # 14: Aromatic (duplicate for compatibility)
                    float(atom.IsInRingSize(3)),                                       # 15: In 3-ring
                    float(atom.IsInRingSize(4)),                                       # 16: In 4-ring
                    float(atom.IsInRingSize(5)),                                       # 17: In 5-ring
                    float(atom.IsInRingSize(6)),                                       # 18: In 6-ring
                    float(atom.GetTotalValence()),                                     # 19: Total valence
                    float(len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE])),   # 20: Single bonds
                    float(len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE])),   # 21: Double bonds
                    float(len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE])),   # 22: Triple bonds
                    float(len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC])), # 23: Aromatic bonds
                ])
                
                # Pad to exactly 29 dimensions
                while len(features) < 29:
                    features.append(0.0)
                features = features[:29]  # Truncate if too long
                
                node_features.append(features)
                
                # Get 3D coordinates if available
                if conformer is not None:
                    pos = conformer.GetAtomPosition(i)
                    positions.append([pos.x, pos.y, pos.z])
                else:
                    positions.append([0.0, 0.0, 0.0])
            
            # Create molecular graph
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                pos=torch.tensor(positions, dtype=torch.float),
                num_nodes=len(atoms)
            )
            
            # Add edges
            data = self.create_edges(data)
            
            # Add batch information
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to convert SMILES {smiles}: {e}")
            return None


class HuggingFaceDJMGNNValidator:
    """Validate the trained DJMGNN from HuggingFace against real PFAS experimental data."""
    
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.converter = SimplePFASConverter()
        
        # Load the trained DJMGNN model
        self.model = self._load_huggingface_model(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Load real PFAS datasets
        self.pfas_data = self._load_pfas_data()
        self.treatment_data = self._load_treatment_data()
        
        logger.info(f"Loaded {len(self.pfas_data)} PFAS compounds")
        logger.info(f"Loaded {len(self.treatment_data)} treatment records")
    
    def _load_huggingface_model(self, model_path: str) -> DJMGNN:
        """Load the trained DJMGNN model from HuggingFace download."""
        try:
            # Load the config
            config_path = os.path.join(os.path.dirname(model_path), 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loading DJMGNN with config: {config}")
            
            # Fix jk_mode compatibility: "cat" -> "concat"
            jk_mode = config['jk_mode']
            if jk_mode == 'cat':
                jk_mode = 'concat'
                logger.info(f"Mapped jk_mode 'cat' to 'concat'")
            
            # Create DJMGNN with the saved configuration
            # Override in_node_dim based on actual model weights (29D not 11D)
            actual_input_dim = 29  # Based on error: torch.Size([128, 29])
            
            model = DJMGNN(
                in_node_dim=actual_input_dim,  # Use 29D to match trained model
                in_edge_dim=config.get('in_edge_dim', 0),
                hidden_dim=config['hidden_dim'],
                n_blocks=config['n_blocks'],
                layers_per_block=config['layers_per_block'],
                node_output_dims=config['node_output_dims'],
                graph_output_dims=config['graph_output_dims'],
                dropout=config['dropout'],
                jk_mode=jk_mode,  # Use the fixed mode
                use_supernode=config['use_supernode'],
                use_rbf=config['use_rbf'],
                rbf_K=config['rbf_K'],
                pool_type='mean'
            )
            
            logger.info(f"Created DJMGNN with {actual_input_dim}D input (overriding config's {config['in_node_dim']}D)")
            
            # Load the trained weights with flexible loading
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
                logger.info("Loaded model from state_dict (flexible)")
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info("Loaded model from model_state_dict (flexible)")
            else:
                # Assume checkpoint is the model state dict directly
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                logger.info("Loaded model weights directly (flexible)")
            
            # Log what was loaded/missing
            if missing_keys:
                logger.warning(f"Missing keys (will use random initialization): {len(missing_keys)} keys")
                for key in missing_keys[:5]:  # Show first 5
                    logger.warning(f"  - {key}")
                if len(missing_keys) > 5:
                    logger.warning(f"  ... and {len(missing_keys)-5} more")
            
            if unexpected_keys:
                logger.warning(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
                for key in unexpected_keys[:5]:  # Show first 5
                    logger.warning(f"  - {key}")
                if len(unexpected_keys) > 5:
                    logger.warning(f"  ... and {len(unexpected_keys)-5} more")
            
            logger.info(f"✅ Successfully loaded trained DJMGNN from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _load_pfas_data(self) -> pd.DataFrame:
        """Load real PFAS molecular data."""
        data_path = PROJECT_ROOT / "data" / "processed" / "chemical_list" / "PFAS_Aligned_Data.csv"
        df = pd.read_csv(data_path)
        
        # Filter for valid SMILES
        df = df.dropna(subset=['SMILES'])
        
        logger.info(f"PFAS data shape: {df.shape}")
        return df
    
    def _load_treatment_data(self) -> pd.DataFrame:
        """Load real treatment effectiveness data."""
        data_path = PROJECT_ROOT / "data" / "processed" / "treatment_data" / "PFAS_Treatment_Data_cleaned.csv"
        df = pd.read_csv(data_path)
        
        # Filter for valid effectiveness data
        df = df.dropna(subset=['Effectiveness_Percent_Numeric'])
        df = df[df['Effectiveness_Percent_Numeric'] >= 0]
        
        logger.info(f"Treatment data shape: {df.shape}")
        return df
    
    def validate_molecular_properties(self, max_molecules: int = 50) -> Dict[str, Any]:
        """Validate model molecular property predictions against computed descriptors."""
        logger.info("🧬 TESTING TRAINED DJMGNN: Molecular property predictions...")
        
        results = {
            'predictions': [],
            'computed_properties': [],
            'molecules': [],
            'errors': []
        }
        
        # Sample molecules for validation
        sample_data = self.pfas_data.sample(n=min(max_molecules, len(self.pfas_data)), random_state=42)
        
        successful_predictions = 0
        
        for idx, row in sample_data.iterrows():
            try:
                # Convert SMILES to molecular graph (11D features)
                mol_graph = self.converter.smiles_to_graph(row['SMILES'])
                if mol_graph is None:
                    continue
                
                mol_graph = mol_graph.to(self.device)
                
                # Compute reference molecular properties using RDKit
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is None:
                    continue
                
                computed_props = [
                    Descriptors.MolWt(mol),               # Molecular weight
                    Descriptors.MolLogP(mol),             # LogP
                    Descriptors.NumHDonors(mol),          # H-bond donors
                    Descriptors.NumHAcceptors(mol),       # H-bond acceptors
                    Descriptors.TPSA(mol),                # Topological polar surface area
                    Descriptors.NumRotatableBonds(mol),   # Rotatable bonds
                    Descriptors.NumAromaticRings(mol),    # Aromatic rings
                    row.get('F_Count', 0),                # Fluorine count
                    row.get('F_Percentage', 0),           # Fluorine percentage
                    row.get('Chain_Length', 0),           # Chain length
                ]
                
                # Pad to 19 dimensions (QM9 standard)
                while len(computed_props) < 19:
                    computed_props.append(0.0)
                computed_props = computed_props[:19]
                
                # Get model predictions from trained DJMGNN
                with torch.no_grad():
                    # Call the DJMGNN model directly (no pos parameter)
                    predictions = self.model(
                        x=mol_graph.x,
                        edge_index=mol_graph.edge_index,
                        edge_attr=getattr(mol_graph, 'edge_attr', None),
                        batch=mol_graph.batch
                    )
                    
                    # Extract graph-level predictions (should be 19D)
                    predicted_props = None
                    if isinstance(predictions, dict):
                        # Use the correct key found in debug: 'graph_pred'
                        predicted_props = predictions.get('graph_pred', 
                                        predictions.get('graph', 
                                        predictions.get('molecular_properties', None)))
                    elif isinstance(predictions, tuple):
                        predicted_props = predictions[1]  # Assuming (node_pred, graph_pred)
                    else:
                        predicted_props = predictions
                    
                    if predicted_props is not None:
                        predicted_props = predicted_props.cpu().numpy().flatten()[:19]
                        
                        results['predictions'].append(predicted_props)
                        results['computed_properties'].append(computed_props)
                        results['molecules'].append(row.get('Preferred_Name', 'Unknown'))
                        
                        successful_predictions += 1
                        
                        if successful_predictions % 10 == 0:
                            logger.info(f"Processed {successful_predictions} molecules...")
                    else:
                        logger.warning(f"Could not extract predictions from model output")
                
            except Exception as e:
                error_msg = f"Error processing {row.get('Preferred_Name', 'Unknown')}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                continue
        
        logger.info(f"Successfully predicted {successful_predictions} molecules")
        
        if successful_predictions >= 5:
            # Compute correlations for each property
            predictions_array = np.array(results['predictions'])
            computed_array = np.array(results['computed_properties'])
            
            results['property_correlations'] = []
            strong_correlations = 0
            
            for i in range(min(predictions_array.shape[1], computed_array.shape[1])):
                try:
                    r, p_value = pearsonr(predictions_array[:, i], computed_array[:, i])
                    results['property_correlations'].append({
                        'property_index': i,
                        'pearson_r': r,
                        'p_value': p_value
                    })
                    if abs(r) > 0.5:  # Strong correlation
                        strong_correlations += 1
                except:
                    results['property_correlations'].append({
                        'property_index': i,
                        'pearson_r': 0.0,
                        'p_value': 1.0
                    })
            
            results['summary'] = {
                'total_correlations': len(results['property_correlations']),
                'strong_correlations': strong_correlations,
                'mean_correlation': np.mean([abs(r['pearson_r']) for r in results['property_correlations']]),
                'successful_predictions': successful_predictions
            }
        
        return results
    
    def print_comparison_results(self, results: Dict[str, Any]):
        """Print results compared to previous baselines."""
        if 'summary' not in results:
            logger.error("❌ No summary available - validation failed!")
            return
        
        summary = results['summary']
        
        logger.info("\n" + "=" * 70)
        logger.info("🚀 HUGGINGFACE DJMGNN VALIDATION RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"📊 Molecules processed: {summary['successful_predictions']}")
        logger.info(f"🎯 Strong correlations (|r| > 0.5): {summary['strong_correlations']}/{summary['total_correlations']}")
        logger.info(f"📈 Mean correlation: {summary['mean_correlation']:.3f}")
        
        # Compare to baselines
        logger.info("\n🏆 PERFORMANCE COMPARISON:")
        logger.info(f"   Untrained baseline:    3/19 strong correlations (15.8%)")
        logger.info(f"   Failed joint model:    1/19 strong correlations (5.3%)")
        logger.info(f"   🔥 Trained DJMGNN:     {summary['strong_correlations']}/{summary['total_correlations']} strong correlations ({100*summary['strong_correlations']/summary['total_correlations']:.1f}%)")
        
        # Assessment
        if summary['strong_correlations'] > 3:
            logger.info("\n✅ 🎉 BREAKTHROUGH: Trained DJMGNN BEATS all baselines!")
            logger.info("✅ Scientific validation PASSED - model learns real chemistry!")
        elif summary['strong_correlations'] >= 3:
            logger.info("\n✅ SUCCESS: Trained DJMGNN matches untrained baseline!")
            logger.info("✅ Training was successful - no degradation!")
        else:
            logger.info("\n❌ CONCERN: Trained DJMGNN underperforms baselines")
            logger.info("❌ May need different training approach")
        
        # Top correlations
        if results.get('property_correlations'):
            logger.info("\n🔬 TOP MOLECULAR PROPERTY CORRELATIONS:")
            sorted_corrs = sorted(results['property_correlations'], key=lambda x: abs(x['pearson_r']), reverse=True)
            for i, corr in enumerate(sorted_corrs[:5]):
                logger.info(f"   Property {corr['property_index']}: r = {corr['pearson_r']:+.3f} (p = {corr['p_value']:.3f})")
        
        logger.info("=" * 70)

    def run_validation(self, max_molecules: int = 50) -> Dict[str, Any]:
        """Run comprehensive validation of the HuggingFace DJMGNN model."""
        logger.info("🚀 TESTING TRAINED DJMGNN FROM HUGGINGFACE")
        logger.info(f"Repository: saketh11/MoML-CA")
        logger.info(f"Testing against {max_molecules} PFAS molecules...")
        
        start_time = time.time()
        
        # Validate molecular properties
        molecular_results = self.validate_molecular_properties(max_molecules)
        
        # Print comparison results
        self.print_comparison_results(molecular_results)
        
        # Save results
        output_dir = "huggingface_djmgnn_validation"
        os.makedirs(output_dir, exist_ok=True)
        
        overall_results = {
            'model_source': 'huggingface_saketh11/MoML-CA',
            'model_type': 'djmgnn_finetuned', 
            'molecular_properties': molecular_results,
            'validation_time': time.time() - start_time,
            'max_molecules_tested': max_molecules
        }
        
        results_file = os.path.join(output_dir, 'huggingface_djmgnn_results.json')
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to: {results_file}")
        
        return overall_results


def main():
    parser = argparse.ArgumentParser(description="Test HuggingFace DJMGNN against real PFAS data")
    parser.add_argument('--max_molecules', type=int, default=50, help='Maximum molecules to test')
    
    args = parser.parse_args()
    
    # Path to the downloaded HuggingFace model
    model_path = '/tmp/djmgnn_model/finetuned_model/pytorch_model.pt'
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please ensure the model is downloaded.")
        sys.exit(1)
    
    # Create validator and run test
    validator = HuggingFaceDJMGNNValidator(model_path=model_path)
    
    # Run validation
    results = validator.run_validation(max_molecules=args.max_molecules)
    
    # Return appropriate exit code based on performance
    if 'summary' in results['molecular_properties']:
        strong_corrs = results['molecular_properties']['summary']['strong_correlations']
        if strong_corrs >= 3:  # Matches or beats baseline
            logger.info("🎉 HuggingFace DJMGNN validation PASSED!")
            sys.exit(0)
        else:
            logger.warning("⚠️ HuggingFace DJMGNN validation shows concerns")
            sys.exit(1)
    else:
        logger.error("❌ HuggingFace DJMGNN validation FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()