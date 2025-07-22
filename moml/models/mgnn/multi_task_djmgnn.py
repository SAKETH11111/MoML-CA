"""
moml/models/mgnn/multi_task_djmgnn.py

Multi-Task Dense Junction Molecular Graph Neural Network

This module extends the successful DJMGNN architecture to support multiple tasks:
1. Molecular property prediction (19D) - Already working at 95% accuracy
2. Force field parameter prediction - Required for MD simulations
3. Treatment efficacy prediction - For PFAS-specific applications

This simplified approach avoids the complexity issues that caused the Joint MGNN to fail.
"""

import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from moml.models.mgnn.djmgnn import DJMGNN

logger = logging.getLogger(__name__)


class ForceFieldHead(nn.Module):
    """
    Specialized head for predicting force field parameters from node embeddings.
    
    Predicts:
    - Partial charges (per atom)
    - Bond parameters (per bond)
    - Angle parameters (per angle)
    - Dihedral parameters (per dihedral)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        # Partial charge prediction (per atom)
        self.charge_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Single charge value per atom
        )
        
        # Bond parameter prediction (requires pair of atom embeddings)
        self.bond_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # k_bond, r_eq
        )
        
        # Angle parameter prediction (requires triplet of atom embeddings)
        self.angle_mlp = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # k_angle, theta_eq
        )
        
        # Dihedral parameter prediction (requires quadruplet of atom embeddings)
        self.dihedral_mlp = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # V_n, gamma, n
        )
    
    def forward(
        self, 
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        angle_indices: Optional[torch.Tensor] = None,
        dihedral_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict force field parameters from node embeddings.
        
        Args:
            node_embeddings: Node feature embeddings [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            angle_indices: Angle atom triplets [3, num_angles] (optional)
            dihedral_indices: Dihedral atom quadruplets [4, num_dihedrals] (optional)
            
        Returns:
            Dictionary containing:
            - partial_charges: [num_nodes, 1]
            - bond_params: [num_edges, 2] (k_bond, r_eq)
            - angle_params: [num_angles, 2] (k_angle, theta_eq) if angle_indices provided
            - dihedral_params: [num_dihedrals, 3] (V_n, gamma, n) if dihedral_indices provided
        """
        results = {}
        
        # Predict partial charges
        partial_charges = self.charge_head(node_embeddings)
        results['partial_charges'] = torch.tanh(partial_charges) * 0.5  # Constrain to [-0.5, 0.5]
        
        # Predict bond parameters
        if edge_index.numel() > 0:
            src_embeddings = node_embeddings[edge_index[0]]
            dst_embeddings = node_embeddings[edge_index[1]]
            bond_features = torch.cat([src_embeddings, dst_embeddings], dim=-1)
            bond_params = self.bond_mlp(bond_features)
            # Ensure positive values with appropriate scales
            results['bond_params'] = torch.stack([
                F.softplus(bond_params[:, 0]) * 500 + 100,  # k_bond: 100-600 kcal/mol/A^2
                F.sigmoid(bond_params[:, 1]) * 2 + 0.5      # r_eq: 0.5-2.5 Angstroms
            ], dim=-1)
        
        # Predict angle parameters if indices provided
        if angle_indices is not None and angle_indices.numel() > 0:
            atom1 = node_embeddings[angle_indices[0]]
            atom2 = node_embeddings[angle_indices[1]]
            atom3 = node_embeddings[angle_indices[2]]
            angle_features = torch.cat([atom1, atom2, atom3], dim=-1)
            angle_params = self.angle_mlp(angle_features)
            results['angle_params'] = torch.stack([
                F.softplus(angle_params[:, 0]) * 100 + 20,     # k_angle: 20-120 kcal/mol/rad^2
                F.sigmoid(angle_params[:, 1]) * 3.14159        # theta_eq: 0-π radians
            ], dim=-1)
        
        # Predict dihedral parameters if indices provided
        if dihedral_indices is not None and dihedral_indices.numel() > 0:
            atom1 = node_embeddings[dihedral_indices[0]]
            atom2 = node_embeddings[dihedral_indices[1]]
            atom3 = node_embeddings[dihedral_indices[2]]
            atom4 = node_embeddings[dihedral_indices[3]]
            dihedral_features = torch.cat([atom1, atom2, atom3, atom4], dim=-1)
            dihedral_params = self.dihedral_mlp(dihedral_features)
            results['dihedral_params'] = torch.stack([
                F.softplus(dihedral_params[:, 0]) * 5,         # V_n: 0-5 kcal/mol
                torch.tanh(dihedral_params[:, 1]) * 3.14159,   # gamma: -π to π
                torch.round(F.sigmoid(dihedral_params[:, 2]) * 6) + 1  # n: 1-6 (periodicity)
            ], dim=-1)
        
        return results


class MultiTaskDJMGNN(nn.Module):
    """
    Multi-task extension of DJMGNN for comprehensive molecular modeling.
    
    This model extends the successful DJMGNN architecture with additional
    task-specific heads while maintaining the core architecture that achieved
    95% accuracy on molecular property prediction.
    """
    
    def __init__(
        self,
        # Base DJMGNN configuration
        djmgnn_config: Dict,
        # Multi-task configuration
        predict_force_field: bool = True,
        predict_treatment_efficacy: bool = True,
        force_field_hidden_dim: int = 128,
        treatment_hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize Multi-Task DJMGNN.
        
        Args:
            djmgnn_config: Configuration for base DJMGNN model
            predict_force_field: Whether to predict force field parameters
            predict_treatment_efficacy: Whether to predict treatment efficacy
            force_field_hidden_dim: Hidden dimension for force field head
            treatment_hidden_dim: Hidden dimension for treatment efficacy head
            dropout: Dropout rate for task heads
        """
        super().__init__()
        
        # Initialize base DJMGNN
        self.djmgnn = DJMGNN(**djmgnn_config)
        
        # Get output dimensions from DJMGNN
        self.node_output_dim = djmgnn_config.get('node_output_dims', djmgnn_config['hidden_dim'])
        self.graph_output_dim = djmgnn_config.get('graph_output_dims', djmgnn_config['hidden_dim'])
        
        # Task flags
        self.predict_force_field = predict_force_field
        self.predict_treatment_efficacy = predict_treatment_efficacy
        
        # Initialize task-specific heads
        if self.predict_force_field:
            self.force_field_head = ForceFieldHead(
                input_dim=self.node_output_dim,
                hidden_dim=force_field_hidden_dim,
                dropout=dropout
            )
        
        if self.predict_treatment_efficacy:
            self.treatment_head = nn.Sequential(
                nn.Linear(self.graph_output_dim, treatment_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(treatment_hidden_dim, 1),
                nn.Sigmoid()  # Efficacy as percentage [0, 1]
            )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        dist: Optional[torch.Tensor] = None,
        angle_indices: Optional[torch.Tensor] = None,
        dihedral_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task model.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge attributes
            batch: Batch indices
            dist: Edge distances
            angle_indices: Indices for angle calculations
            dihedral_indices: Indices for dihedral calculations
            
        Returns:
            Dictionary containing predictions for all tasks
        """
        # Get base DJMGNN predictions
        djmgnn_out = self.djmgnn(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            dist=dist
        )
        
        # Start with base predictions
        results = {
            'molecular_properties': djmgnn_out['graph_pred'],  # 19D properties
            'node_embeddings': djmgnn_out['node_pred'],       # For downstream use
            'energy': djmgnn_out['energy_pred']               # Energy prediction
        }
        
        # Add force field predictions if enabled
        if self.predict_force_field and djmgnn_out['node_pred'].numel() > 0:
            ff_predictions = self.force_field_head(
                node_embeddings=djmgnn_out['node_pred'],
                edge_index=edge_index,
                angle_indices=angle_indices,
                dihedral_indices=dihedral_indices
            )
            results.update(ff_predictions)
        
        # Add treatment efficacy prediction if enabled
        if self.predict_treatment_efficacy and djmgnn_out['graph_pred'].numel() > 0:
            # Use graph-level features for treatment efficacy
            results['treatment_efficacy'] = self.treatment_head(djmgnn_out['graph_pred'])
        
        return results
    
    def compute_multi_task_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            task_weights: Optional task-specific weights
            
        Returns:
            Total loss and individual task losses
        """
        if task_weights is None:
            task_weights = {
                'molecular_properties': 1.0,
                'partial_charges': 2.0,      # Important for MD
                'bond_params': 1.5,
                'angle_params': 1.5,
                'dihedral_params': 1.0,
                'treatment_efficacy': 2.0,
                'energy': 0.5
            }
        
        total_loss = 0.0
        task_losses = {}
        
        # Molecular properties loss (MSE)
        if 'molecular_properties' in targets and 'molecular_properties' in predictions:
            loss = F.mse_loss(predictions['molecular_properties'], targets['molecular_properties'])
            task_losses['molecular_properties'] = loss
            total_loss += task_weights.get('molecular_properties', 1.0) * loss
        
        # Force field parameter losses
        if 'partial_charges' in targets and 'partial_charges' in predictions:
            loss = F.mse_loss(predictions['partial_charges'], targets['partial_charges'])
            task_losses['partial_charges'] = loss
            total_loss += task_weights.get('partial_charges', 1.0) * loss
        
        if 'bond_params' in targets and 'bond_params' in predictions:
            loss = F.mse_loss(predictions['bond_params'], targets['bond_params'])
            task_losses['bond_params'] = loss
            total_loss += task_weights.get('bond_params', 1.0) * loss
        
        if 'angle_params' in targets and 'angle_params' in predictions:
            loss = F.mse_loss(predictions['angle_params'], targets['angle_params'])
            task_losses['angle_params'] = loss
            total_loss += task_weights.get('angle_params', 1.0) * loss
        
        if 'dihedral_params' in targets and 'dihedral_params' in predictions:
            loss = F.mse_loss(predictions['dihedral_params'], targets['dihedral_params'])
            task_losses['dihedral_params'] = loss
            total_loss += task_weights.get('dihedral_params', 1.0) * loss
        
        # Treatment efficacy loss (BCE since it's [0,1])
        if 'treatment_efficacy' in targets and 'treatment_efficacy' in predictions:
            loss = F.binary_cross_entropy(predictions['treatment_efficacy'], targets['treatment_efficacy'])
            task_losses['treatment_efficacy'] = loss
            total_loss += task_weights.get('treatment_efficacy', 1.0) * loss
        
        # Energy loss
        if 'energy' in targets and 'energy' in predictions:
            loss = F.mse_loss(predictions['energy'], targets['energy'])
            task_losses['energy'] = loss
            total_loss += task_weights.get('energy', 1.0) * loss
        
        return total_loss, task_losses


def create_multi_task_djmgnn(base_config: Dict, multi_task_config: Optional[Dict] = None) -> MultiTaskDJMGNN:
    """
    Factory function to create a multi-task DJMGNN model.
    
    Args:
        base_config: Configuration for base DJMGNN
        multi_task_config: Additional multi-task configuration
        
    Returns:
        Configured MultiTaskDJMGNN instance
    """
    if multi_task_config is None:
        multi_task_config = {
            'predict_force_field': True,
            'predict_treatment_efficacy': True,
            'force_field_hidden_dim': 128,
            'treatment_hidden_dim': 64,
            'dropout': 0.1
        }
    
    return MultiTaskDJMGNN(
        djmgnn_config=base_config,
        **multi_task_config
    )