#!/usr/bin/env python3
"""
validate_hmgnn.py

Test a trained HMGNN model against real PFAS experimental data.

This script validates whether a trained HMGNN model performs better than
the established baselines.

Usage:
    python scripts/validate_hmgnn.py --model_path checkpoints/hmgnn_production_best.pt --max_molecules 50
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.feature_transforms import CreateEdges
from moml.models.mgnn.hmgnn import HMGNN
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# Suppress RDKit deprecation warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePFASConverter:
    """Convert PFAS SMILES to molecular graphs with 29D node features."""
    
    def __init__(self):
        self.create_edges = CreateEdges()
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to a molecular graph."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                AllChem.UFFOptimizeMolecule(mol)

            atoms = mol.GetAtoms()
            conformer = mol.GetConformer() if mol.GetNumConformers() > 0 else None
            
            node_features = []
            positions = []
            
            for i, atom in enumerate(atoms):
                features = [
                    float(atom.GetAtomicNum()), float(atom.GetDegree()), float(atom.GetFormalCharge()),
                    float(atom.GetHybridization()), float(atom.GetIsAromatic()), float(atom.GetTotalNumHs()),
                    float(atom.GetMass()), float(atom.IsInRing()), float(atom.GetImplicitValence()),
                    float(atom.GetExplicitValence()), float(atom.GetNumRadicalElectrons())
                ]
                features.extend([
                    float(atom.GetHybridization() == Chem.HybridizationType.SP),
                    float(atom.GetHybridization() == Chem.HybridizationType.SP2),
                    float(atom.GetHybridization() == Chem.HybridizationType.SP3),
                    float(atom.GetIsAromatic()), float(atom.IsInRingSize(3)), float(atom.IsInRingSize(4)),
                    float(atom.IsInRingSize(5)), float(atom.IsInRingSize(6)), float(atom.GetTotalValence())
                ])
                features.extend([
                    float(len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.SINGLE])),
                    float(len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE])),
                    float(len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE])),
                    float(len([b for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC]))
                ])

                while len(features) < 29: features.append(0.0)
                node_features.append(features[:29])
                
                if conformer:
                    pos = conformer.GetAtomPosition(i)
                    positions.append([pos.x, pos.y, pos.z])
                else:
                    positions.append([0.0, 0.0, 0.0])
            
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                pos=torch.tensor(positions, dtype=torch.float),
                num_nodes=len(atoms)
            )
            data = self.create_edges(data)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            return data
            
        except Exception as e:
            logger.warning(f"Failed to convert SMILES {smiles}: {e}")
            return None


class HMGNNValidator:
    """Validate a trained HMGNN against real PFAS experimental data."""
    
    def __init__(self, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.converter = SimplePFASConverter()
        
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.pfas_data = self._load_pfas_data()
        logger.info(f"Loaded {len(self.pfas_data)} PFAS compounds")

    def _load_model(self, model_path: str) -> HMGNN:
        """Load a trained HMGNN model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint.get('config', self._get_default_hmgnn_config())
            
            logger.info(f"Loading HMGNN with config: {config}")
            model = HMGNN(**config)
            
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"✅ Successfully loaded trained HMGNN from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def _get_default_hmgnn_config(self):
        """Returns a default config dictionary for HMGNN."""
        return {
            'scale_dims': [29, 29, 29],
            'hidden_dim': 128,
            'n_blocks': 3,
            'layers_per_block': 3,
            'node_out_dim': 3,
            'graph_out_dim': 19,
            'dropout': 0.2,
            'pool_type': 'mean'
        }

    def _load_pfas_data(self) -> pd.DataFrame:
        """Load real PFAS molecular data."""
        data_path = PROJECT_ROOT / "data" / "processed" / "chemical_list" / "PFAS_Aligned_Data.csv"
        df = pd.read_csv(data_path)
        return df.dropna(subset=['SMILES'])

    def validate_molecular_properties(self, max_molecules: int = 50) -> Dict[str, Any]:
        """Validate model's molecular property predictions."""
        logger.info("🧬 TESTING TRAINED HMGNN: Molecular property predictions...")
        results = {'predictions': [], 'computed_properties': [], 'molecules': [], 'errors': []}
        sample_data = self.pfas_data.sample(n=min(max_molecules, len(self.pfas_data)), random_state=42)
        
        for _, row in sample_data.iterrows():
            try:
                mol_graph = self.converter.smiles_to_graph(row['SMILES'])
                if mol_graph is None: continue
                
                mol_graph = mol_graph.to(self.device)
                
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is None: continue
                
                computed_props = [
                    Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol), row.get('F_Count', 0),
                    row.get('F_Percentage', 0), row.get('Chain_Length', 0)
                ]
                while len(computed_props) < 19: computed_props.append(0.0)
                
                with torch.no_grad():
                    scale_data = [{'x': mol_graph.x, 'edge_index': mol_graph.edge_index, 'batch': mol_graph.batch}]*3
                    predictions = self.model(scale_data=scale_data)
                    predicted_props = predictions.get('graph_pred')

                    if predicted_props is not None:
                        results['predictions'].append(predicted_props.cpu().numpy().flatten()[:19])
                        results['computed_properties'].append(computed_props[:19])
                        results['molecules'].append(row.get('Preferred_Name', 'Unknown'))
                    else:
                        logger.warning("Could not extract predictions.")
            except Exception as e:
                logger.error(f"Error processing {row.get('Preferred_Name', 'Unknown')}: {e}")
        
        if results['predictions']:
            predictions_array = np.array(results['predictions'])
            computed_array = np.array(results['computed_properties'])
            results['property_correlations'] = []
            for i in range(min(predictions_array.shape[1], computed_array.shape[1])):
                r, p_value = pearsonr(predictions_array[:, i], computed_array[:, i])
                results['property_correlations'].append({'property_index': i, 'pearson_r': r, 'p_value': p_value})
            
            results['summary'] = {
                'strong_correlations': sum(1 for r in results['property_correlations'] if abs(r['pearson_r']) > 0.5)
            }
        return results

    def run_validation(self, max_molecules: int = 50):
        """Run comprehensive validation."""
        logger.info(f"🧪 Validating HMGNN against {max_molecules} PFAS molecules...")
        molecular_results = self.validate_molecular_properties(max_molecules)
        self.print_results(molecular_results)

        output_dir = "hmgnn_validation"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'hmgnn_results.json'), 'w') as f:
            json.dump(molecular_results, f, indent=2, default=str)
        logger.info(f"📁 Results saved to: {output_dir}/hmgnn_results.json")
    
    def print_results(self, results):
        if 'summary' not in results:
            logger.error("❌ Validation failed!")
            return
        
        summary = results['summary']
        logger.info("\n" + "="*70)
        logger.info("📄 HMGNN VALIDATION RESULTS")
        logger.info("="*70)
        logger.info(f"🎯 Strong correlations (|r| > 0.5): {summary['strong_correlations']}/19")
        
        if summary['strong_correlations'] >= 3:
            logger.info("✅ SUCCESS: HMGNN meets or beats baseline performance!")
        else:
            logger.info("⚠️ CONCERN: HMGNN underperforms baseline.")

        if results.get('property_correlations'):
            logger.info("\n🔬 Top Correlations:")
            sorted_corrs = sorted(results['property_correlations'], key=lambda x: abs(x['pearson_r']), reverse=True)
            for corr in sorted_corrs[:5]:
                logger.info(f"   Property {corr['property_index']}: r = {corr['pearson_r']:+.3f}")
        logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(description="Test trained HMGNN against real PFAS data")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained HMGNN model checkpoint')
    parser.add_argument('--max_molecules', type=int, default=50, help='Maximum molecules to test')
    args = parser.parse_args()
    
    validator = HMGNNValidator(model_path=args.model_path)
    validator.run_validation(max_molecules=args.max_molecules)


if __name__ == '__main__':
    main()