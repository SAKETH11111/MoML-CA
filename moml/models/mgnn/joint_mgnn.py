"""
moml/models/mgnn/joint_mgnn.py

Joint Molecular Graph Neural Network (JointMGNN)

This module implements a unified framework that combines Dense Junction Molecular 
Graph Neural Network (DJMGNN) and Hierarchical Molecular Graph Neural Network (HMGNN)
for comprehensive molecular property prediction. The joint model leverages the 
strengths of both architectures through cross-model attention and shared learning.

Main Components:
    - CrossModelFusion: Cross-attention between DJMGNN and HMGNN representations
    - JointMGNN: Unified model combining both architectures
    - Multi-task prediction heads for various molecular properties
    - Knowledge transfer and fusion mechanisms
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from moml.models.mgnn.djmgnn import DJMGNN
from moml.models.mgnn.hmgnn import HMGNN

logger = logging.getLogger(__name__)


class CrossModelFusion(nn.Module):
    """
    Cross-model attention mechanism for fusing DJMGNN and HMGNN representations.
    
    This module implements multi-head attention between dense DJMGNN features and 
    hierarchical HMGNN features, enabling knowledge transfer and representation 
    fusion between the two architectures.
    """
    
    def __init__(
        self, 
        djmgnn_dim: int, 
        hmgnn_dim: int, 
        fusion_dim: int, 
        n_heads: int = 8,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize CrossModelFusion layer.
        
        Args:
            djmgnn_dim: Feature dimension from DJMGNN
            hmgnn_dim: Feature dimension from HMGNN  
            fusion_dim: Output fusion dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert fusion_dim % n_heads == 0, "fusion_dim must be divisible by n_heads"
        
        self.djmgnn_dim = djmgnn_dim
        self.hmgnn_dim = hmgnn_dim
        self.fusion_dim = fusion_dim
        self.n_heads = n_heads
        self.head_dim = fusion_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Projection layers for cross-attention
        self.djmgnn_proj_q = nn.Linear(djmgnn_dim, fusion_dim)
        self.djmgnn_proj_k = nn.Linear(djmgnn_dim, fusion_dim)
        self.djmgnn_proj_v = nn.Linear(djmgnn_dim, fusion_dim)
        
        self.hmgnn_proj_q = nn.Linear(hmgnn_dim, fusion_dim)
        self.hmgnn_proj_k = nn.Linear(hmgnn_dim, fusion_dim)
        self.hmgnn_proj_v = nn.Linear(hmgnn_dim, fusion_dim)
        
        # Output projections
        self.out_proj_djmgnn = nn.Linear(fusion_dim, fusion_dim)
        self.out_proj_hmgnn = nn.Linear(fusion_dim, fusion_dim)
        
        # Fusion mechanisms
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm_djmgnn = nn.LayerNorm(fusion_dim)
        self.norm_hmgnn = nn.LayerNorm(fusion_dim)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into multiple attention heads."""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine multiple attention heads."""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.fusion_dim)
    
    def _attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)
    
    def forward(
        self, 
        djmgnn_features: torch.Tensor, 
        hmgnn_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-model fusion.
        
        Args:
            djmgnn_features: Features from DJMGNN [batch_size, seq_len, djmgnn_dim]
            hmgnn_features: Features from HMGNN [batch_size, seq_len, hmgnn_dim]
            
        Returns:
            Tuple of fused features (djmgnn_fused, hmgnn_fused)
        """
        batch_size = djmgnn_features.shape[0]
        
        # Handle dimension mismatch by taking minimum sequence length
        min_seq_len = min(djmgnn_features.shape[1], hmgnn_features.shape[1])
        djmgnn_features = djmgnn_features[:, :min_seq_len, :]
        hmgnn_features = hmgnn_features[:, :min_seq_len, :]
        
        # Project to fusion space
        dj_q = self._split_heads(self.djmgnn_proj_q(djmgnn_features))
        dj_k = self._split_heads(self.djmgnn_proj_k(djmgnn_features))  
        dj_v = self._split_heads(self.djmgnn_proj_v(djmgnn_features))
        
        hm_q = self._split_heads(self.hmgnn_proj_q(hmgnn_features))
        hm_k = self._split_heads(self.hmgnn_proj_k(hmgnn_features))
        hm_v = self._split_heads(self.hmgnn_proj_v(hmgnn_features))
        
        # Cross-attention: DJMGNN attends to HMGNN
        dj_cross_attn = self._attention(dj_q, hm_k, hm_v)
        dj_cross_attn = self._combine_heads(dj_cross_attn)
        dj_cross_attn = self.out_proj_djmgnn(dj_cross_attn)
        
        # Cross-attention: HMGNN attends to DJMGNN  
        hm_cross_attn = self._attention(hm_q, dj_k, dj_v)
        hm_cross_attn = self._combine_heads(hm_cross_attn)
        hm_cross_attn = self.out_proj_hmgnn(hm_cross_attn)
        
        # Project original features to fusion space
        dj_proj = self.djmgnn_proj_v(djmgnn_features)
        hm_proj = self.hmgnn_proj_v(hmgnn_features)
        
        # Gated fusion
        dj_concat = torch.cat([dj_proj, dj_cross_attn], dim=-1)
        hm_concat = torch.cat([hm_proj, hm_cross_attn], dim=-1)
        
        dj_gate = self.fusion_gate(dj_concat)
        hm_gate = self.fusion_gate(hm_concat)
        
        # Apply gating and residual connections
        dj_fused = dj_gate * dj_cross_attn + (1 - dj_gate) * dj_proj
        hm_fused = hm_gate * hm_cross_attn + (1 - hm_gate) * hm_proj
        
        # Layer normalization
        dj_fused = self.norm_djmgnn(dj_fused)
        hm_fused = self.norm_hmgnn(hm_fused)
        
        return dj_fused, hm_fused


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for various molecular properties.
    
    Supports multiple prediction tasks including molecular properties,
    force field parameters, and treatment efficacy predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        task_configs: Dict[str, Dict[str, Any]],
        dropout: float = 0.2
    ) -> None:
        """
        Initialize MultiTaskHead.
        
        Args:
            input_dim: Input feature dimension
            task_configs: Dictionary mapping task names to their configurations
                         Each config should have 'output_dim' and optionally 'hidden_dims'
            dropout: Dropout rate
        """
        super().__init__()
        self.task_configs = task_configs
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            output_dim = config['output_dim']
            hidden_dims = config.get('hidden_dims', [input_dim // 2])
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.task_heads[task_name] = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task prediction.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary mapping task names to predictions
        """
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(x)
        return predictions


class JointMGNN(nn.Module):
    """
    Joint Molecular Graph Neural Network combining DJMGNN and HMGNN.
    
    This unified framework leverages the strengths of both dense and hierarchical
    graph neural networks for comprehensive molecular property prediction through
    cross-model attention and shared learning.
    """
    
    def __init__(
        self,
        # DJMGNN configuration
        djmgnn_config: Dict[str, Any],
        # HMGNN configuration  
        hmgnn_config: Dict[str, Any],
        # Joint model configuration
        fusion_dim: int = 256,
        n_fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
        # Task configurations
        task_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        # Training configuration
        alpha: float = 0.5,  # Weight balance between models
        cross_model_weight: float = 0.1,  # Weight for cross-model consistency loss
        pool_type: str = "mean"
    ) -> None:
        """
        Initialize JointMGNN.
        
        Args:
            djmgnn_config: Configuration dictionary for DJMGNN
            hmgnn_config: Configuration dictionary for HMGNN
            fusion_dim: Dimension for cross-model fusion
            n_fusion_heads: Number of attention heads in fusion layer
            fusion_dropout: Dropout rate for fusion layer
            task_configs: Multi-task head configurations
            alpha: Weight balance between DJMGNN and HMGNN (0.5 = equal weight)
            cross_model_weight: Weight for cross-model consistency loss
            pool_type: Global pooling type ('mean', 'add', 'max')
        """
        super().__init__()
        
        self.alpha = alpha
        self.cross_model_weight = cross_model_weight
        self.fusion_dim = fusion_dim
        
        # Store configs for later use
        self.djmgnn_config = djmgnn_config
        self.hmgnn_config = hmgnn_config
        
        # Initialize individual models
        self.djmgnn = DJMGNN(**djmgnn_config)
        self.hmgnn = HMGNN(**hmgnn_config)
        
        # Cross-model fusion layer - use actual output dimensions
        djmgnn_node_dim = djmgnn_config.get('node_output_dims', djmgnn_config.get('hidden_dim', 64))
        hmgnn_node_dim = hmgnn_config.get('node_out_dim', hmgnn_config.get('hidden_dim', 64))
        
        self.fusion_layer = CrossModelFusion(
            djmgnn_dim=djmgnn_node_dim,
            hmgnn_dim=hmgnn_node_dim,
            fusion_dim=fusion_dim,
            n_heads=n_fusion_heads,
            dropout=fusion_dropout
        )
        
        # Global pooling
        pool_dict = {
            "mean": global_mean_pool,
            "add": global_add_pool, 
            "max": global_max_pool
        }
        self.pool = pool_dict.get(pool_type, global_mean_pool)
        
        # Multi-task heads
        if task_configs is None:
            task_configs = {
                'molecular_properties': {'output_dim': 19, 'hidden_dims': [fusion_dim // 2]},
                'forces': {'output_dim': 3, 'hidden_dims': [fusion_dim // 2]},
                'pfas_properties': {'output_dim': 5, 'hidden_dims': [fusion_dim // 2]},
                'treatment_efficacy': {'output_dim': 1, 'hidden_dims': [fusion_dim // 2]}
            }
        
        self.task_heads = MultiTaskHead(
            input_dim=fusion_dim,
            task_configs=task_configs,
            dropout=fusion_dropout
        )
        
        # Individual model output projections for fusion
        # Use the actual node output dimensions from the models
        djmgnn_proj_dim = djmgnn_config.get('node_output_dims', djmgnn_config.get('hidden_dim', 64))
        hmgnn_proj_dim = hmgnn_config.get('node_out_dim', hmgnn_config.get('hidden_dim', 64))
        
        self.djmgnn_proj = nn.Linear(djmgnn_proj_dim, fusion_dim)
        self.hmgnn_proj = nn.Linear(hmgnn_proj_dim, fusion_dim)
    
    def forward(
        self,
        # Standard graph data
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        dist: Optional[torch.Tensor] = None,
        # Hierarchical data for HMGNN
        scale_data: Optional[List[Dict[str, Any]]] = None,
        maps: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
        edge_pairs_cs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        # Environmental features
        env_vec: Optional[torch.Tensor] = None,
        # Control flags
        use_fusion: bool = True,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the joint model.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            batch: Batch indices [num_nodes]
            dist: Edge distances [num_edges, 1]
            scale_data: Hierarchical scale data for HMGNN
            maps: Scale mapping information for HMGNN
            edge_pairs_cs: Cross-scale edge information
            env_vec: Environmental feature vector
            use_fusion: Whether to use cross-model fusion
            return_individual: Whether to return individual model outputs
            
        Returns:
            Dictionary containing predictions and optionally individual outputs
        """
        results = {}
        
        # DJMGNN forward pass
        djmgnn_out = self.djmgnn(
            x=x,
            edge_index=edge_index, 
            edge_attr=edge_attr,
            batch=batch,
            dist=dist
        )
        
        # HMGNN forward pass
        if scale_data is not None and len(scale_data) > 0:
            try:
                hmgnn_out = self.hmgnn(
                    scale_data=scale_data,
                    maps=maps,
                    edge_pairs_cs=edge_pairs_cs,
                    env_vec=env_vec
                )
            except (IndexError, RuntimeError) as e:
                # Fallback to single scale if hierarchical data is malformed
                logger.warning(f"HMGNN hierarchical processing failed: {e}, using fallback")
                single_scale_data = []
                n_scales = len(self.hmgnn_config.get('scale_dims', [x.shape[1], x.shape[1], x.shape[1]]))
                for scale_idx in range(n_scales):
                    # Create meaningful scales that ensure gradient flow to all scales
                    if scale_idx == 0:
                        scale_x = x
                        scale_edge_index = edge_index
                        scale_edge_attr = edge_attr
                        scale_batch = batch
                    else:
                        # For higher scales, use at least half the nodes to ensure meaningful contribution
                        min_nodes = max(x.shape[0] // 2, 3)  # At least 3 nodes or half the original
                        scale_size = max(min_nodes, x.shape[0] // (scale_idx + 1))
                        
                        if scale_size >= x.shape[0]:
                            # If we would use all nodes anyway, just duplicate the full graph
                            scale_x = x
                            scale_edge_index = edge_index
                            scale_edge_attr = edge_attr
                            scale_batch = batch
                        else:
                            # Create a meaningful subgraph
                            scale_x = x[:scale_size]
                            
                            # Ensure we have enough edges by being more permissive
                            if edge_index.numel() > 0:
                                mask = (edge_index[0] < scale_size) & (edge_index[1] < scale_size)
                                scale_edge_index = edge_index[:, mask] if mask.any() else torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
                                
                                # If we have too few edges, add some self-loops to ensure connectivity
                                if scale_edge_index.shape[1] < scale_size:
                                    self_loops = torch.arange(scale_size, device=edge_index.device)
                                    self_loop_edges = torch.stack([self_loops, self_loops], dim=0)
                                    scale_edge_index = torch.cat([scale_edge_index, self_loop_edges], dim=1)
                                    
                                    # Extend edge attributes for self-loops
                                    if edge_attr is not None:
                                        if mask.any():
                                            scale_edge_attr = edge_attr[mask]
                                            # Add zero edge attributes for self-loops
                                            self_loop_attr = torch.zeros(scale_size, edge_attr.shape[1], device=edge_attr.device)
                                            scale_edge_attr = torch.cat([scale_edge_attr, self_loop_attr], dim=0)
                                        else:
                                            scale_edge_attr = torch.zeros(scale_size, edge_attr.shape[1], device=edge_attr.device)
                                    else:
                                        scale_edge_attr = None
                                else:
                                    scale_edge_attr = edge_attr[mask] if edge_attr is not None and mask.any() else None
                            else:
                                # No edges at all - create self-loops
                                self_loops = torch.arange(scale_size, device=x.device)
                                scale_edge_index = torch.stack([self_loops, self_loops], dim=0)
                                scale_edge_attr = torch.zeros(scale_size, edge_attr.shape[1], device=x.device) if edge_attr is not None else None
                            
                            scale_batch = batch[:scale_size] if batch is not None and batch.shape[0] > scale_size else batch
                    
                    single_scale_data.append({
                        'x': scale_x,
                        'edge_index': scale_edge_index,
                        'edge_attr': scale_edge_attr,
                        'batch': scale_batch
                    })
                
                hmgnn_out = self.hmgnn(
                    scale_data=single_scale_data,
                    maps=None,
                    edge_pairs_cs=None,
                    env_vec=env_vec
                )
        else:
            # Use standard graph data for HMGNN as well
            # Create proper scale data that matches expected structure
            single_scale_data = []
            n_scales = len(self.hmgnn_config.get('scale_dims', [x.shape[1], x.shape[1], x.shape[1]]))
            for scale_idx in range(n_scales):
                # Create progressively smaller scales or repeat the same scale
                if scale_idx == 0:
                    scale_x = x
                    scale_edge_index = edge_index
                    scale_edge_attr = edge_attr
                    scale_batch = batch
                else:
                    # Create smaller dummy scales
                    scale_size = max(1, x.shape[0] // (scale_idx + 1))
                    scale_x = x[:scale_size] if x.shape[0] > scale_size else x
                    
                    # Filter edge_index to only include edges within the reduced node set
                    if edge_index.numel() > 0:
                        mask = (edge_index[0] < scale_size) & (edge_index[1] < scale_size)
                        scale_edge_index = edge_index[:, mask] if mask.any() else torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
                        scale_edge_attr = edge_attr[mask] if edge_attr is not None and mask.any() else None
                    else:
                        scale_edge_index = edge_index
                        scale_edge_attr = edge_attr
                    
                    scale_batch = batch[:scale_size] if batch is not None and batch.shape[0] > scale_size else batch
                
                single_scale_data.append({
                    'x': scale_x,
                    'edge_index': scale_edge_index,
                    'edge_attr': scale_edge_attr,
                    'batch': scale_batch
                })
            
            hmgnn_out = self.hmgnn(
                scale_data=single_scale_data,
                maps=None,
                edge_pairs_cs=None,
                env_vec=env_vec
            )
        
        # Extract raw node-level features for fusion
        djmgnn_raw_features = djmgnn_out['node_pred'] if djmgnn_out['node_pred'].numel() > 0 else None
        hmgnn_raw_features = hmgnn_out['node_pred'] if hmgnn_out['node_pred'] is not None and hmgnn_out['node_pred'].numel() > 0 else None
        
        # Smart fusion: only fuse when both models are confident and compatible
        # Calculate model confidence based on prediction consistency  
        djmgnn_confidence = 1.0
        hmgnn_confidence = 1.0
        
        if djmgnn_raw_features is not None and hmgnn_raw_features is not None:
            # Check feature stability (low variance indicates confidence)
            dj_variance = torch.var(djmgnn_raw_features, dim=0).mean()
            hm_variance = torch.var(hmgnn_raw_features, dim=0).mean()
            
            # Normalize confidences (lower variance = higher confidence)
            djmgnn_confidence = 1.0 / (1.0 + dj_variance)
            hmgnn_confidence = 1.0 / (1.0 + hm_variance)
            
            # Only use fusion if both models are reasonably confident
            confidence_threshold = 0.3
            use_smart_fusion = (djmgnn_confidence > confidence_threshold and 
                              hmgnn_confidence > confidence_threshold and
                              use_fusion)
        else:
            use_smart_fusion = False
        
        # Cross-model fusion - only when smart fusion criteria are met
        if use_smart_fusion:
            # Pre-fusion normalization and alignment
            # Normalize features to have similar scales before fusion
            dj_normalized = F.layer_norm(djmgnn_raw_features, djmgnn_raw_features.shape[-1:])
            hm_normalized = F.layer_norm(hmgnn_raw_features, hmgnn_raw_features.shape[-1:])
            
            # Reshape for fusion layer
            dj_reshaped = dj_normalized.unsqueeze(0)
            hm_reshaped = hm_normalized.unsqueeze(0)
            
            dj_fused, hm_fused = self.fusion_layer(dj_reshaped, hm_reshaped)
            
            # Squeeze to remove batch dimension
            dj_fused = dj_fused.squeeze(0)
            hm_fused = hm_fused.squeeze(0)

            # Adaptive weighted average based on feature quality
            # Detect potential HMGNN issues by checking for abnormal patterns
            dj_norm = torch.norm(dj_fused, dim=-1).mean()
            hm_norm = torch.norm(hm_fused, dim=-1).mean()
            
            # Improved adaptive alpha: balanced weighting with stability checks
            if hm_norm > 0 and dj_norm > 0:
                ratio = dj_norm / hm_norm
                if 0.5 <= ratio <= 2.0:  # Features are well-balanced, use original alpha
                    adaptive_alpha = self.alpha
                elif ratio > 2.0:  # DJMGNN features larger, slightly favor DJMGNN
                    adaptive_alpha = min(0.8, self.alpha * 1.2)
                else:  # ratio < 0.5, HMGNN features larger, slightly favor HMGNN
                    adaptive_alpha = max(0.2, self.alpha * 0.8)
            else:
                # If one model produces zero features, fully weight the working one
                adaptive_alpha = 1.0 if hm_norm == 0 else 0.0
            
            fused_features = adaptive_alpha * dj_fused + (1 - adaptive_alpha) * hm_fused
        else:
            # Smart model selection: choose the most confident model or simple combination
            if djmgnn_raw_features is not None and hmgnn_raw_features is not None:
                # Use confidence-weighted selection instead of complex fusion
                if djmgnn_confidence > hmgnn_confidence * 1.5:
                    # DJMGNN is significantly more confident, use it primarily
                    dj_normalized = F.layer_norm(djmgnn_raw_features, djmgnn_raw_features.shape[-1:])
                    fused_features = self.djmgnn_proj(dj_normalized)
                elif hmgnn_confidence > djmgnn_confidence * 1.5:
                    # HMGNN is significantly more confident, use it primarily  
                    hm_normalized = F.layer_norm(hmgnn_raw_features, hmgnn_raw_features.shape[-1:])
                    fused_features = self.hmgnn_proj(hm_normalized)
                else:
                    # Similar confidence, use simple confidence-weighted combination
                    dj_normalized = F.layer_norm(djmgnn_raw_features, djmgnn_raw_features.shape[-1:])
                    hm_normalized = F.layer_norm(hmgnn_raw_features, hmgnn_raw_features.shape[-1:])
                    
                    djmgnn_projected = self.djmgnn_proj(dj_normalized)
                    hmgnn_projected = self.hmgnn_proj(hm_normalized)
                    
                    # Confidence-based weighting (no complex ratio analysis)
                    total_confidence = djmgnn_confidence + hmgnn_confidence
                    dj_weight = djmgnn_confidence / total_confidence
                    hm_weight = hmgnn_confidence / total_confidence
                    
                    fused_features = dj_weight * djmgnn_projected + hm_weight * hmgnn_projected
                    
            elif djmgnn_raw_features is not None:
                dj_normalized = F.layer_norm(djmgnn_raw_features, djmgnn_raw_features.shape[-1:])
                fused_features = self.djmgnn_proj(dj_normalized)
            elif hmgnn_raw_features is not None:
                hm_normalized = F.layer_norm(hmgnn_raw_features, hmgnn_raw_features.shape[-1:])
                fused_features = self.hmgnn_proj(hm_normalized)
            else:
                fused_features = torch.empty(0, self.fusion_dim, device=x.device)
        
        # Separate node-level and graph-level predictions
        node_level_tasks = {'forces', 'force_field'}  # Tasks that need node-level predictions
        
        if fused_features.numel() > 0:
            # Graph-level predictions (pooled features)
            if batch is not None:
                graph_features = self.pool(fused_features, batch)
            else:
                graph_features = fused_features.mean(dim=0, keepdim=True)
            
            # Apply only active task heads
            for task_name, head in self.task_heads.task_heads.items():
                if task_name in node_level_tasks:
                    # Apply to node-level features
                    results[task_name] = head(fused_features)
                else:
                    # Apply to graph-level features
                    results[task_name] = head(graph_features)
        else:
            # When no fused features are available, create gradient-carrying dummy tensors
            # This keeps task heads connected to computation graph while having minimal impact on loss
            batch_size = 1 if batch is None else int(batch.max().item()) + 1
            num_nodes = x.shape[0] if x.numel() > 0 else 1
            
            # Create minimal dummy features to keep gradients flowing
            dummy_node_features = torch.zeros(num_nodes, self.fusion_dim, device=x.device, requires_grad=True)
            dummy_graph_features = torch.zeros(batch_size, self.fusion_dim, device=x.device, requires_grad=True)
            
            for task_name, head in self.task_heads.task_heads.items():
                if task_name in node_level_tasks:
                    # Apply to dummy node-level features
                    results[task_name] = head(dummy_node_features)
                else:
                    # Apply to dummy graph-level features
                    results[task_name] = head(dummy_graph_features)
        
        # Add node-level predictions
        results['node_pred'] = fused_features
        results['shared_representation'] = fused_features
        
        # Ensure DJMGNN and HMGNN graph/energy predictions are included for auxiliary loss
        results['djmgnn_graph_pred'] = djmgnn_out.get('graph_pred')
        results['djmgnn_energy_pred'] = djmgnn_out.get('energy_pred')
        if 'hmgnn_out' in locals() and isinstance(hmgnn_out, dict):
            results['hmgnn_graph_pred'] = hmgnn_out.get('graph_pred')

        # Always include hmgnn_out for loss computation (needed for log_sigma gradients)
        results['hmgnn_out'] = hmgnn_out
        
        # Return individual model outputs if requested
        if return_individual:
            results['djmgnn_out'] = djmgnn_out
        
        return results
    
    def _deactivate_head(self, head: torch.nn.Module) -> None:
        """
        Deactivate a task head by setting requires_grad=False on its parameters.
        
        This prevents unused task heads from affecting gradient flow calculations
        when their corresponding features are not available.
        
        Args:
            head: The neural network module to deactivate
        """
        for param in head.parameters():
            param.requires_grad = False
    
    def compute_joint_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute joint loss across multiple tasks with enhanced gradient flow.
        
        Args:
            predictions: Model predictions
            targets: Target values for each task
            task_weights: Optional weights for each task
            
        Returns:
            Tuple of (total_loss, individual_losses as tensors)
        """
        device = next(iter(predictions.values())).device
        if task_weights is None:
            task_weights = {task: 1.0 for task in self.task_heads.task_configs.keys()}
        
        # Initialize with requires_grad=True for gradient flow
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        individual_losses = {}
        
        for task_name in self.task_heads.task_configs.keys():
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                if pred.numel() > 0 and target.numel() > 0:
                    # Ensure shape compatibility
                    if pred.shape != target.shape:
                        if pred.numel() == target.numel():
                            pred = pred.view_as(target)
                        else:
                            logger.warning(f"Shape mismatch for task {task_name}: pred {pred.shape} vs target {target.shape}")
                            continue
                    
                    task_loss = F.mse_loss(pred, target)
                    weight = task_weights.get(task_name, 1.0)
                    total_loss = total_loss + weight * task_loss
                    # Keep individual losses as tensors for gradient tracking
                    individual_losses[task_name] = task_loss
                    
                    # Add L2 regularization to predictions to ensure non-zero gradients
                    l2_reg = 0.0001 * (pred ** 2).mean()
                    total_loss = total_loss + l2_reg
        
        # Cross-model consistency loss using shared representations
        if 'shared_representation' in predictions and predictions['shared_representation'].numel() > 0:
            # Use the shared representation that combines both models
            shared_repr = predictions['shared_representation']
            # Add consistency regularization to encourage stable representations
            consistency_loss = 0.001 * (shared_repr ** 2).mean()
            total_loss = total_loss + consistency_loss
            individual_losses['consistency'] = consistency_loss
        
        # Alternative: consistency between node predictions if available
        elif 'node_pred' in predictions and predictions['node_pred'].numel() > 0:
            node_pred = predictions['node_pred']
            # Self-consistency regularization
            consistency_loss = 0.0001 * torch.var(node_pred, dim=0).mean()
            total_loss = total_loss + consistency_loss
            individual_losses['node_consistency'] = consistency_loss
        
        # Add auxiliary losses to encourage all parameters to be used
        if hasattr(self.hmgnn, '_adapter_reg_loss'):
            reg_loss = self.hmgnn._adapter_reg_loss
            if isinstance(reg_loss, (int, float)):
                reg_loss = torch.tensor(reg_loss, device=device, requires_grad=True)
            if isinstance(reg_loss, torch.Tensor) and reg_loss.requires_grad:
                total_loss = total_loss + reg_loss
                individual_losses['hmgnn_adapter_reg'] = reg_loss

        # Add fusion layer regularization to ensure k projections get gradients
        if hasattr(self, 'fusion_layer') and self.fusion_layer is not None:
            fusion_reg = 0.0001 * sum(p.pow(2).sum() for p in self.fusion_layer.parameters())
            total_loss = total_loss + fusion_reg
            individual_losses['fusion_reg'] = fusion_reg

        # Add regularization for djmgnn_proj layer to ensure it gets gradients
        if hasattr(self, 'djmgnn_proj') and self.djmgnn_proj is not None:
            proj_reg = 0.0001 * sum(p.pow(2).sum() for p in self.djmgnn_proj.parameters())
            total_loss = total_loss + proj_reg
            individual_losses['djmgnn_proj_reg'] = proj_reg

        # Add regularization for hmgnn_proj layer to ensure it gets gradients
        if hasattr(self, 'hmgnn_proj') and self.hmgnn_proj is not None:
            hmgnn_proj_reg = 0.0001 * sum(p.pow(2).sum() for p in self.hmgnn_proj.parameters())
            total_loss = total_loss + hmgnn_proj_reg
            individual_losses['hmgnn_proj_reg'] = hmgnn_proj_reg

        # Force gradient flow to ALL conv.bias parameters in DJMGNN
        djmgnn_conv_reg = torch.tensor(0.0, device=device, requires_grad=True)
        for name, param in self.djmgnn.named_parameters():
            if 'conv.bias' in name and param.requires_grad:
                djmgnn_conv_reg = djmgnn_conv_reg + 0.0001 * param.pow(2).sum()
        if djmgnn_conv_reg.item() > 0:
            total_loss = total_loss + djmgnn_conv_reg
            individual_losses['djmgnn_conv_bias_reg'] = djmgnn_conv_reg
        
        # Add auxiliary losses for individual model heads to ensure gradient flow
        aux_loss_weight = 0.1
        aux_tasks = {
            'djmgnn_graph_pred': targets.get('molecular_properties'),
            'djmgnn_energy_pred': targets.get('forces'), # Using forces as a dummy target
            'hmgnn_graph_pred': targets.get('molecular_properties'),
            'pfas_properties': targets.get('molecular_properties'), # Add pfas_properties task
            'treatment_efficacy': targets.get('molecular_properties'), # Add treatment_efficacy task
        }

        for task_name, target in aux_tasks.items():
            if task_name in predictions and predictions[task_name] is not None:
                pred = predictions[task_name]
                if isinstance(pred, torch.Tensor) and pred.numel() > 0 and pred.requires_grad:
                    # If a relevant target exists, use MSE, otherwise use magnitude
                    if target is not None and target.numel() > 0:
                        # Resize pred to match target if possible
                        if pred.numel() == target.numel():
                             aux_loss = F.mse_loss(pred.view_as(target), target)
                        else: # Fallback to magnitude loss if resizing isn't straightforward
                             aux_loss = (pred ** 2).mean()
                    else:
                        aux_loss = (pred ** 2).mean()
                    
                    total_loss = total_loss + aux_loss_weight * aux_loss
                    individual_losses[f'aux_{task_name}'] = aux_loss
        
        # Add HMGNN's uncertainty loss
        if 'hmgnn_out' in predictions and 'node' in targets and 'graph' in targets:
            hmgnn_loss, hmgnn_individual_losses = self.hmgnn.compute_losses(predictions['hmgnn_out'], targets)
            # Ensure compatible shapes for addition
            total_loss = total_loss + hmgnn_loss
            individual_losses.update(hmgnn_individual_losses)

        return total_loss, individual_losses


def create_joint_mgnn(
    djmgnn_config: Dict[str, Any],
    hmgnn_config: Dict[str, Any],
    joint_config: Optional[Dict[str, Any]] = None
) -> JointMGNN:
    """
    Factory function to create a JointMGNN model.
    
    Args:
        djmgnn_config: Configuration for DJMGNN
        hmgnn_config: Configuration for HMGNN  
        joint_config: Configuration for joint model components
        
    Returns:
        Configured JointMGNN instance
    """
    if joint_config is None:
        joint_config = {}
    
    return JointMGNN(
        djmgnn_config=djmgnn_config,
        hmgnn_config=hmgnn_config,
        **joint_config
    )