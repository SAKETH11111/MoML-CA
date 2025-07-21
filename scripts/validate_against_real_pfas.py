#!/usr/bin/env python3
"""
validate_against_real_pfas.py

CRITICAL SCIENTIFIC VALIDATION: Test our Joint MGNN against real PFAS experimental data.

This script validates whether our model learns meaningful PFAS relationships by comparing
predictions against experimental treatment effectiveness and molecular properties.

Key Questions:
1. Can we predict real PFAS treatment effectiveness?
2. Do our PFAS property predictions correlate with experimental data?
3. Are we learning chemistry or fitting noise?

Usage:
    python scripts/validate_against_real_pfas.py
"""

import argparse
import json
import logging
import os
import sys
import time
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

from moml.data.feature_transforms import CreateEdges, FeaturizeNodes
from moml.models.mgnn import create_joint_mgnn
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from rdkit.Geometry import Point3D

# Suppress RDKit deprecation warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PFASMoleculeConverter:
    """Convert PFAS SMILES to molecular graphs compatible with our model."""
    
    def __init__(self):
        self.create_edges = CreateEdges()
        self.featurize_nodes = FeaturizeNodes()
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to molecular graph with 3D coordinates."""
        try:
            # Parse SMILES and add hydrogens
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for realistic 3D geometry
            mol = Chem.AddHs(mol)
            
            # Generate 3D conformer
            try:
                # Try to embed molecule in 3D space
                AllChem.EmbedMolecule(mol, randomSeed=42)
                
                # Optimize geometry using MMFF force field
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except:
                    # Fallback to UFF if MMFF fails
                    AllChem.UFFOptimizeMolecule(mol)
                    
            except Exception as embed_error:
                logger.warning(f"Failed to generate 3D conformer for {smiles}: {embed_error}")
                # Try without conformer generation (will use 2D connectivity only)
                mol = Chem.RemoveHs(mol)  # Remove Hs if 3D generation failed
            
            # Get atom features and coordinates
            atoms = mol.GetAtoms()
            conformer = mol.GetConformer() if mol.GetNumConformers() > 0 else None
            
            node_features = []
            positions = []
            
            for i, atom in enumerate(atoms):
                # Basic atomic features (matching training featurization)
                features = [
                    float(atom.GetAtomicNum()),           # Atomic number
                    float(atom.GetDegree()),              # Degree
                    float(atom.GetFormalCharge()),        # Formal charge
                    float(atom.GetHybridization()),       # Hybridization
                    float(atom.GetIsAromatic()),          # Aromaticity
                    float(atom.GetTotalNumHs()),          # Total hydrogens
                ]
                
                # Add more chemical features to reach 29 dimensions
                features.extend([
                    float(atom.GetMass()),                # Atomic mass
                    float(atom.IsInRing()),               # In ring
                    float(atom.GetImplicitValence()),     # Implicit valence
                    float(atom.GetExplicitValence()),     # Explicit valence
                ])
                
                # Pad to match expected dimension (29)
                while len(features) < 29:
                    features.append(0.0)
                
                node_features.append(features[:29])  # Truncate if too long
                
                # Get 3D coordinates if available
                if conformer is not None:
                    pos = conformer.GetAtomPosition(i)
                    positions.append([pos.x, pos.y, pos.z])
                else:
                    # Use dummy coordinates if no 3D structure
                    positions.append([0.0, 0.0, 0.0])
            
            # Create molecular graph with 3D coordinates
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                pos=torch.tensor(positions, dtype=torch.float),  # 3D coordinates
                num_nodes=len(atoms)
            )
            
            # Add edges using the same transform as training (now with 3D coordinates)
            data = self.create_edges(data)
            
            # Add batch information for single molecule
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to convert SMILES {smiles}: {e}")
            return None


class RealPFASValidator:
    """Validate Joint MGNN against real PFAS experimental data."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.converter = PFASMoleculeConverter()
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            # Create model first
            self.model = self._create_validation_model()
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded trained model from checkpoint: {model_path}")
            else:
                self.model = checkpoint
                logger.info(f"Loaded model directly from {model_path}")
        else:
            # Create model with same config as validation
            self.model = self._create_validation_model()
            logger.info("Created new model for testing")
        
        self.model.eval()
        self.model = self.model.to(self.device)  # Ensure model is on correct device
        
        # Load real PFAS datasets
        self.pfas_data = self._load_pfas_data()
        self.treatment_data = self._load_treatment_data()
        
        logger.info(f"Loaded {len(self.pfas_data)} PFAS compounds")
        logger.info(f"Loaded {len(self.treatment_data)} treatment records")
    
    def _create_validation_model(self):
        """Create model with same configuration as production training."""
        # Match production model architecture exactly
        djmgnn_config = {
            'in_node_dim': 29,
            'hidden_dim': 128,  # Increased from validation
            'n_blocks': 3,      # Increased for production
            'layers_per_block': 4,
            'node_output_dims': 3,
            'graph_output_dims': 19,
            'dropout': 0.15,
            'pool_type': 'mean'
        }
        
        hmgnn_config = {
            'scale_dims': [29, 29, 29],
            'hidden_dim': 128,
            'n_blocks': 3,
            'layers_per_block': 3,
            'node_out_dim': 3,
            'graph_out_dim': 19,
            'dropout': 0.15,
            'pool_type': 'mean'
        }
        
        joint_config = {
            'fusion_dim': 256,
            'n_fusion_heads': 8,
            'alpha': 0.5
        }
        
        return create_joint_mgnn(
            djmgnn_config=djmgnn_config,
            hmgnn_config=hmgnn_config,
            joint_config=joint_config
        )
    
    def _load_pfas_data(self) -> pd.DataFrame:
        """Load real PFAS molecular data."""
        data_path = PROJECT_ROOT / "data" / "processed" / "chemical_list" / "PFAS_Aligned_Data.csv"
        df = pd.read_csv(data_path)
        
        # Filter for valid SMILES and complete data
        df = df.dropna(subset=['SMILES', 'Effectiveness_Percent_Numeric'])
        
        logger.info(f"PFAS data shape: {df.shape}")
        logger.info(f"Available treatment processes: {df['Treatment_Process'].value_counts().head()}")
        
        return df
    
    def _load_treatment_data(self) -> pd.DataFrame:
        """Load real treatment effectiveness data."""
        data_path = PROJECT_ROOT / "data" / "processed" / "treatment_data" / "PFAS_Treatment_Data_cleaned.csv"
        df = pd.read_csv(data_path)
        
        # Filter for valid effectiveness data
        df = df.dropna(subset=['Effectiveness_Percent_Numeric'])
        df = df[df['Effectiveness_Percent_Numeric'] >= 0]  # Remove invalid percentages
        
        logger.info(f"Treatment data shape: {df.shape}")
        logger.info(f"Effectiveness range: {df['Effectiveness_Percent_Numeric'].min():.1f} - {df['Effectiveness_Percent_Numeric'].max():.1f}%")
        
        return df
    
    def validate_treatment_effectiveness(self, max_molecules: int = 100) -> Dict[str, Any]:
        """Validate model predictions against real treatment effectiveness data."""
        logger.info("Validating treatment effectiveness predictions...")
        
        results = {
            'predictions': [],
            'experimental': [],
            'molecules': [],
            'treatment_methods': [],
            'errors': []
        }
        
        # Get subset of molecules with both SMILES and treatment data
        logger.info(f"PFAS data shape: {self.pfas_data.shape}, Treatment data shape: {self.treatment_data.shape}")
        logger.info(f"PFAS CASRN sample: {self.pfas_data['CASRN'].head().tolist()}")
        logger.info(f"Treatment CASRN sample: {self.treatment_data['CASRN'].head().tolist()}")
        
        merged_data = self.pfas_data.merge(
            self.treatment_data,
            left_on='CASRN',
            right_on='CASRN',
            how='inner'
        )
        
        logger.info(f"Found {len(merged_data)} molecules with both molecular and treatment data")
        
        if len(merged_data) == 0:
            logger.error("❌ NO MERGED DATA! PFAS and treatment datasets have no overlapping CASRN values!")
            return results
        
        # Debug: Check what columns are available in merged data
        logger.info(f"Available columns in merged data: {list(merged_data.columns)}")
        effectiveness_cols = [col for col in merged_data.columns if 'effectiveness' in col.lower() or 'percent' in col.lower()]
        logger.info(f"Effectiveness-related columns: {effectiveness_cols}")
        
        # Sample subset for validation
        if len(merged_data) > max_molecules:
            merged_data = merged_data.sample(n=max_molecules, random_state=42)
        
        successful_predictions = 0
        
        for idx, row in merged_data.iterrows():
            try:
                logger.info(f"Processing molecule {successful_predictions+1}: {row.get('Preferred_Name', 'Unknown')} - SMILES: {row['SMILES'][:50]}...")
                
                # Convert SMILES to molecular graph
                mol_graph = self.converter.smiles_to_graph(row['SMILES'])
                if mol_graph is None:
                    logger.warning(f"Failed to convert SMILES to graph: {row['SMILES']}")
                    continue
                
                logger.info(f"Successfully created graph with {mol_graph.num_nodes} nodes and {mol_graph.edge_index.shape[1]} edges")
                
                mol_graph = mol_graph.to(self.device)
                
                # Get model predictions - handle joint model properly
                with torch.no_grad():
                    # Check if this is a joint model that needs scale_data
                    if hasattr(self.model, 'djmgnn') and hasattr(self.model, 'hmgnn'):
                        # Create hierarchical scale data for joint model
                        scale_data = [{
                            'x': mol_graph.x,
                            'edge_index': mol_graph.edge_index,
                            'edge_attr': getattr(mol_graph, 'edge_attr', None),
                            'batch': mol_graph.batch
                        }]
                        
                        # Add simplified scales for HMGNN
                        for _ in range(2):  # Additional scales
                            scale_data.append({
                                'x': torch.empty(0, mol_graph.x.shape[1], device=mol_graph.x.device),
                                'edge_index': torch.empty(2, 0, dtype=torch.long, device=mol_graph.x.device),
                                'edge_attr': None,
                                'batch': torch.empty(0, dtype=torch.long, device=mol_graph.x.device)
                            })
                        
                        outputs = self.model(
                            x=mol_graph.x,
                            edge_index=mol_graph.edge_index,
                            edge_attr=getattr(mol_graph, 'edge_attr', None),
                            batch=mol_graph.batch,
                            scale_data=scale_data,
                            use_fusion=True
                        )
                    else:
                        # Individual model
                        outputs = self.model(
                            x=mol_graph.x,
                            edge_index=mol_graph.edge_index,
                            edge_attr=getattr(mol_graph, 'edge_attr', None),
                            batch=mol_graph.batch
                        )
                
                # Extract treatment efficacy prediction
                if 'treatment_efficacy' in outputs:
                    predicted_efficacy = outputs['treatment_efficacy'].cpu().numpy().item()
                    
                    # Handle different column names for experimental effectiveness
                    if 'Effectiveness_Percent_Numeric' in row:
                        experimental_efficacy = row['Effectiveness_Percent_Numeric']
                    elif 'Effectiveness_Percent_Numeric_x' in row:
                        experimental_efficacy = row['Effectiveness_Percent_Numeric_x']
                    elif 'Effectiveness_Percent_Numeric_y' in row:
                        experimental_efficacy = row['Effectiveness_Percent_Numeric_y']
                    else:
                        logger.warning(f"No effectiveness data found for {row.get('Preferred_Name', 'Unknown')}")
                        continue
                    
                    results['predictions'].append(predicted_efficacy)
                    results['experimental'].append(experimental_efficacy)
                    results['molecules'].append(row['Preferred_Name'])
                    results['treatment_methods'].append(row['Treatment_Process'])
                    
                    successful_predictions += 1
                    
                    if successful_predictions % 10 == 0:
                        logger.info(f"Processed {successful_predictions} molecules...")
                
            except Exception as e:
                error_msg = f"Error processing {row.get('Preferred_Name', 'Unknown')}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                continue
        
        logger.info(f"Successfully predicted {successful_predictions} molecules")
        
        if successful_predictions > 5:  # Need minimum data for meaningful statistics
            results['statistics'] = self._compute_prediction_statistics(
                results['predictions'], results['experimental']
            )
        
        return results
    
    def validate_molecular_properties(self, max_molecules: int = 100) -> Dict[str, Any]:
        """Validate model molecular property predictions against computed descriptors."""
        logger.info("Validating molecular property predictions...")
        
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
                # Convert SMILES to molecular graph
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
                    row['F_Count'],                       # Fluorine count
                    row['F_Percentage'],                  # Fluorine percentage
                    row['Chain_Length'],                  # Chain length
                ]
                
                # Pad to 19 dimensions (QM9 standard)
                while len(computed_props) < 19:
                    computed_props.append(0.0)
                computed_props = computed_props[:19]
                
                # Get model predictions - handle joint model properly
                with torch.no_grad():
                    # Check if this is a joint model that needs scale_data
                    if hasattr(self.model, 'djmgnn') and hasattr(self.model, 'hmgnn'):
                        # Create hierarchical scale data for joint model
                        scale_data = [{
                            'x': mol_graph.x,
                            'edge_index': mol_graph.edge_index,
                            'edge_attr': getattr(mol_graph, 'edge_attr', None),
                            'batch': mol_graph.batch
                        }]
                        
                        # Add simplified scales for HMGNN
                        for _ in range(2):  # Additional scales
                            scale_data.append({
                                'x': torch.empty(0, mol_graph.x.shape[1], device=mol_graph.x.device),
                                'edge_index': torch.empty(2, 0, dtype=torch.long, device=mol_graph.x.device),
                                'edge_attr': None,
                                'batch': torch.empty(0, dtype=torch.long, device=mol_graph.x.device)
                            })
                        
                        outputs = self.model(
                            x=mol_graph.x,
                            edge_index=mol_graph.edge_index,
                            edge_attr=getattr(mol_graph, 'edge_attr', None),
                            batch=mol_graph.batch,
                            scale_data=scale_data,
                            use_fusion=True
                        )
                    else:
                        # Individual model
                        outputs = self.model(
                            x=mol_graph.x,
                            edge_index=mol_graph.edge_index,
                            edge_attr=getattr(mol_graph, 'edge_attr', None),
                            batch=mol_graph.batch
                        )
                
                # Extract molecular property predictions
                if 'molecular_properties' in outputs:
                    predicted_props = outputs['molecular_properties'].cpu().numpy().flatten()
                    
                    results['predictions'].append(predicted_props)
                    results['computed_properties'].append(computed_props)
                    results['molecules'].append(row['Preferred_Name'])
                    
                    successful_predictions += 1
                    
                    if successful_predictions % 10 == 0:
                        logger.info(f"Processed {successful_predictions} molecules...")
                
            except Exception as e:
                error_msg = f"Error processing {row.get('Preferred_Name', 'Unknown')}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                continue
        
        logger.info(f"Successfully predicted {successful_predictions} molecules")
        
        if successful_predictions > 5:
            # Compute correlations for each property
            predictions_array = np.array(results['predictions'])
            computed_array = np.array(results['computed_properties'])
            
            results['property_correlations'] = []
            for i in range(min(predictions_array.shape[1], computed_array.shape[1])):
                try:
                    r, p_value = pearsonr(predictions_array[:, i], computed_array[:, i])
                    results['property_correlations'].append({
                        'property_index': i,
                        'pearson_r': r,
                        'p_value': p_value
                    })
                except:
                    results['property_correlations'].append({
                        'property_index': i,
                        'pearson_r': 0.0,
                        'p_value': 1.0
                    })
        
        return results
    
    def _compute_prediction_statistics(self, predictions: List[float], experimental: List[float]) -> Dict[str, float]:
        """Compute statistical metrics for predictions vs experimental data."""
        pred_array = np.array(predictions)
        exp_array = np.array(experimental)
        
        # Handle any potential scaling issues
        # Convert model outputs to percentage scale if needed
        pred_max = np.max(np.abs(pred_array))
        exp_max = np.max(np.abs(exp_array))
        
        # Transform model predictions to match experimental scale (0-100%)
        if exp_max > 50:  # Experimental data is in percentage scale
            # Normalize predictions to [0,1] then scale to [0,100]
            pred_min, pred_max_val = np.min(pred_array), np.max(pred_array)
            
            if pred_max_val > pred_min:  # Avoid division by zero
                # Normalize to [0,1] then scale to [0,100]
                pred_array = (pred_array - pred_min) / (pred_max_val - pred_min) * 100
                logger.info(f"Rescaled predictions to [0,100]: range [{np.min(pred_array):.1f}, {np.max(pred_array):.1f}]")
            else:
                # All predictions are the same value - scale to middle of range
                pred_array = np.full_like(pred_array, 50.0)
                logger.info("All predictions identical - set to 50% for scaling")
        
        return {
            'r2_score': r2_score(exp_array, pred_array),
            'pearson_r': pearsonr(pred_array, exp_array)[0],
            'spearman_r': spearmanr(pred_array, exp_array)[0],
            'mae': mean_absolute_error(exp_array, pred_array),
            'rmse': np.sqrt(mean_squared_error(exp_array, pred_array)),
            'mean_prediction': float(np.mean(pred_array)),
            'mean_experimental': float(np.mean(exp_array)),
            'std_prediction': float(np.std(pred_array)),
            'std_experimental': float(np.std(exp_array)),
            'n_samples': len(predictions)
        }
    
    def create_validation_plots(self, treatment_results: Dict, molecular_results: Dict, output_dir: str):
        """Create visualization plots for validation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Treatment effectiveness plot
        if 'statistics' in treatment_results and len(treatment_results['predictions']) > 0:
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(treatment_results['experimental'], treatment_results['predictions'], alpha=0.6)
            plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
            plt.xlabel('Experimental Effectiveness (%)')
            plt.ylabel('Predicted Effectiveness (%)')
            plt.title(f"Treatment Effectiveness\nR² = {treatment_results['statistics']['r2_score']:.3f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Residuals plot
            plt.subplot(1, 2, 2)
            residuals = np.array(treatment_results['predictions']) - np.array(treatment_results['experimental'])
            plt.scatter(treatment_results['experimental'], residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Experimental Effectiveness (%)')
            plt.ylabel('Residuals (%)')
            plt.title('Prediction Residuals')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'treatment_effectiveness_validation.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Molecular properties correlation plot
        if 'property_correlations' in molecular_results:
            correlations = [result['pearson_r'] for result in molecular_results['property_correlations']]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(correlations)), correlations)
            plt.xlabel('Molecular Property Index')
            plt.ylabel('Pearson Correlation (r)')
            plt.title('Molecular Property Prediction Correlations')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='r = 0.5')
            plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='r = 0.8')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'molecular_property_correlations.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_comprehensive_validation(self, max_molecules: int = 100, output_dir: str = "pfas_validation_results") -> Dict[str, Any]:
        """Run comprehensive validation against real PFAS data."""
        logger.info("=" * 60)
        logger.info("CRITICAL SCIENTIFIC VALIDATION")
        logger.info("Testing Joint MGNN against REAL PFAS experimental data")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Validate treatment effectiveness
        logger.info("\n1. TREATMENT EFFECTIVENESS VALIDATION")
        treatment_results = self.validate_treatment_effectiveness(max_molecules)
        
        # Validate molecular properties
        logger.info("\n2. MOLECULAR PROPERTY VALIDATION")
        molecular_results = self.validate_molecular_properties(max_molecules)
        
        # Create plots
        logger.info("\n3. CREATING VALIDATION PLOTS")
        self.create_validation_plots(treatment_results, molecular_results, output_dir)
        
        # Compile overall results
        overall_results = {
            'treatment_effectiveness': treatment_results,
            'molecular_properties': molecular_results,
            'validation_summary': self._create_validation_summary(treatment_results, molecular_results),
            'validation_time': time.time() - start_time
        }
        
        # Save results
        results_file = os.path.join(output_dir, 'real_pfas_validation_results.json')
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        # Print summary
        self._print_validation_summary(overall_results)
        
        return overall_results
    
    def _create_validation_summary(self, treatment_results: Dict, molecular_results: Dict) -> Dict[str, Any]:
        """Create summary of validation results."""
        summary = {
            'treatment_effectiveness': {
                'samples_processed': len(treatment_results['predictions']),
                'errors_encountered': len(treatment_results['errors']),
                'has_valid_statistics': 'statistics' in treatment_results
            },
            'molecular_properties': {
                'samples_processed': len(molecular_results['predictions']),
                'errors_encountered': len(molecular_results['errors']),
                'has_correlations': 'property_correlations' in molecular_results
            }
        }
        
        # Add statistical summaries if available
        if 'statistics' in treatment_results:
            stats = treatment_results['statistics']
            summary['treatment_effectiveness'].update({
                'r2_score': stats['r2_score'],
                'pearson_r': stats['pearson_r'],
                'mae_percent': stats['mae'],
                'prediction_quality': self._assess_prediction_quality(stats['r2_score'], stats['mae'])
            })
        
        if 'property_correlations' in molecular_results:
            correlations = [r['pearson_r'] for r in molecular_results['property_correlations']]
            summary['molecular_properties'].update({
                'mean_correlation': np.mean(correlations),
                'strong_correlations': sum(1 for r in correlations if abs(r) > 0.5),
                'total_properties': len(correlations)
            })
        
        return summary
    
    def _assess_prediction_quality(self, r2: float, mae: float) -> str:
        """Assess the quality of predictions based on statistical metrics."""
        if r2 > 0.8 and mae < 10:
            return "EXCELLENT"
        elif r2 > 0.6 and mae < 20:
            return "GOOD"
        elif r2 > 0.3 and mae < 30:
            return "FAIR"
        else:
            return "POOR"
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print comprehensive validation summary."""
        summary = results['validation_summary']
        
        logger.info("\n" + "=" * 60)
        logger.info("REAL PFAS VALIDATION RESULTS")
        logger.info("=" * 60)
        
        # Treatment effectiveness results
        logger.info("\n🧪 TREATMENT EFFECTIVENESS VALIDATION:")
        treat_summary = summary['treatment_effectiveness']
        logger.info(f"   Molecules Processed: {treat_summary['samples_processed']}")
        logger.info(f"   Errors Encountered: {treat_summary['errors_encountered']}")
        
        if treat_summary['has_valid_statistics']:
            logger.info(f"   R² Score: {treat_summary['r2_score']:.3f}")
            logger.info(f"   Pearson r: {treat_summary['pearson_r']:.3f}")
            logger.info(f"   Mean Absolute Error: {treat_summary['mae_percent']:.1f}%")
            logger.info(f"   Prediction Quality: {treat_summary['prediction_quality']}")
        else:
            logger.info("   ❌ Insufficient data for statistical analysis")
        
        # Molecular properties results
        logger.info("\n🧬 MOLECULAR PROPERTY VALIDATION:")
        mol_summary = summary['molecular_properties']
        logger.info(f"   Molecules Processed: {mol_summary['samples_processed']}")
        logger.info(f"   Errors Encountered: {mol_summary['errors_encountered']}")
        
        if mol_summary['has_correlations']:
            logger.info(f"   Mean Correlation: {mol_summary['mean_correlation']:.3f}")
            logger.info(f"   Strong Correlations (|r| > 0.5): {mol_summary['strong_correlations']}/{mol_summary['total_properties']}")
        else:
            logger.info("   ❌ Insufficient data for correlation analysis")
        
        # Overall assessment
        logger.info("\n🎯 OVERALL SCIENTIFIC VALIDITY:")
        if (treat_summary.get('prediction_quality') in ['GOOD', 'EXCELLENT'] or 
            mol_summary.get('mean_correlation', 0) > 0.3):
            logger.info("   ✅ Model shows meaningful correlation with experimental data")
            logger.info("   ✅ Scientific validation PASSED - safe to scale up training")
        else:
            logger.info("   ❌ Model predictions do not correlate well with experimental data")
            logger.info("   ❌ Scientific validation FAILED - review model architecture before scaling")
        
        logger.info(f"\nValidation completed in {results['validation_time']:.1f} seconds")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate Joint MGNN against real PFAS data")
    parser.add_argument('--model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--max_molecules', type=int, default=100, help='Maximum molecules to test')
    parser.add_argument('--output_dir', type=str, default='pfas_validation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create validator
    validator = RealPFASValidator(model_path=args.model_path)
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation(
        max_molecules=args.max_molecules,
        output_dir=args.output_dir
    )
    
    # Return appropriate exit code
    summary = results['validation_summary']
    treat_quality = summary['treatment_effectiveness'].get('prediction_quality', 'POOR')
    mol_correlation = summary['molecular_properties'].get('mean_correlation', 0.0)
    
    if treat_quality in ['GOOD', 'EXCELLENT'] or mol_correlation > 0.3:
        logger.info("🎉 Validation PASSED - Model is scientifically valid!")
        sys.exit(0)
    else:
        logger.warning("⚠️  Validation FAILED - Model needs improvement before scaling")
        sys.exit(1)


if __name__ == '__main__':
    main()