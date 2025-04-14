"""
Hierarchical Molecular Graph Neural Network

This module implements the HMGNN model architecture which processes multi-scale
molecular representations for enhanced molecular property prediction, with
particular focus on PFAS molecules and force field parameter prediction.

The key components are:
1. Multi-scale graph processing (atom, functional group, structural motifs)
2. Cross-scale information exchange
3. Multi-task learning for different properties and scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from typing import Dict, List, Optional, Any

from moml.models.mgnn.djmgnn import DenseGNNBlock, JKAggregator

class CrossScaleAttention(nn.Module):
    """
    Attention mechanism for cross-scale information exchange.
    Transfers information between different scales of the molecular representation.
    """
    def __init__(self, scale_dims: List[int], hidden_dim: int):
        """
        Initialize the cross-scale attention mechanism.
        
        Args:
            scale_dims: List of feature dimensions for each scale
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        num_scales = len(scale_dims)
        self.scale_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in scale_dims
        ])
        
        # Query, key, value projections for each scale
        self.query_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_scales)
        ])
        self.key_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_scales)
        ])
        self.value_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_scales)
        ])
        
        # Output projections back to original dimensions
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in scale_dims
        ])
        
        # Attention temperature
        self.temperature = hidden_dim ** 0.5
        
        # Scale mappings (will be set by the HMGNN model)
        self.cluster_mappings = None
    
    def set_cluster_mappings(self, mappings: List[Dict[int, int]]):
        """
        Set the mappings between scales.
        
        Args:
            mappings: List of dictionaries mapping node indices from one scale to the next
        """
        self.cluster_mappings = mappings
    
    def forward(self, scale_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of cross-scale attention.
        
        Args:
            scale_features: List of node features for each scale
                Each tensor has shape [num_nodes_i, feature_dim_i]
                
        Returns:
            List of updated features for each scale
        """
        num_scales = len(scale_features)
        
        if self.cluster_mappings is None or len(self.cluster_mappings) < num_scales - 1:
            # If no mappings are available, just return the input features
            return scale_features
        
        # Project features to common dimension
        projected_features = [
            self.scale_projections[i](scale_features[i])
            for i in range(num_scales)
        ]
        
        # Prepare queries, keys, values
        queries = [
            self.query_projections[i](projected_features[i])
            for i in range(num_scales)
        ]
        keys = [
            self.key_projections[i](projected_features[i])
            for i in range(num_scales)
        ]
        values = [
            self.value_projections[i](projected_features[i])
            for i in range(num_scales)
        ]
        
        # Process information flow for each scale
        updated_features = []
        
        # Process each scale
        for scale_idx in range(num_scales):
            scale_query = queries[scale_idx]
            
            # Initialize updated features with self-attention
            scale_output = torch.zeros_like(scale_query)
            
            # Attend to each other scale
            for other_idx in range(num_scales):
                if scale_idx == other_idx:
                    # Self-attention - just add the values directly
                    scale_output = scale_output + values[scale_idx]
                else:
                    # Cross-scale attention
                    # Need to map between scales using cluster_mappings
                    if scale_idx < other_idx:
                        # Upward attention: fine → coarse
                        # Need to aggregate fine-scale queries to match coarse-scale keys
                        aggregated_query = self._aggregate_features(
                            scale_query, scale_idx, other_idx
                        )
                        
                        # Compute attention scores
                        attn_scores = torch.matmul(
                            aggregated_query, keys[other_idx].transpose(-2, -1)
                        ) / self.temperature
                        attn_weights = F.softmax(attn_scores, dim=-1)
                        
                        # Compute attended values
                        attended_values = torch.matmul(attn_weights, values[other_idx])
                        
                        # Distribute back to original scale
                        scale_output = scale_output + self._distribute_features(
                            attended_values, other_idx, scale_idx
                        )
                        
                    else:
                        # Downward attention: coarse → fine
                        # Need to broadcast coarse-scale queries to match fine-scale keys
                        broadcasted_query = self._broadcast_features(
                            scale_query, scale_idx, other_idx
                        )
                        
                        # Compute attention scores
                        attn_scores = torch.matmul(
                            broadcasted_query, keys[other_idx].transpose(-2, -1)
                        ) / self.temperature
                        attn_weights = F.softmax(attn_scores, dim=-1)
                        
                        # Compute attended values
                        attended_values = torch.matmul(attn_weights, values[other_idx])
                        
                        # Apply to original scale
                        scale_output = scale_output + attended_values
            
            # Project back to original dimension
            updated_feature = scale_features[scale_idx] + self.output_projections[scale_idx](scale_output)
            updated_features.append(updated_feature)
        
        return updated_features
    
    def _aggregate_features(
        self, features: torch.Tensor, from_scale: int, to_scale: int
    ) -> torch.Tensor:
        """
        Aggregate features from a finer scale to a coarser scale.
        
        Args:
            features: Node features at the finer scale
            from_scale: Index of the finer scale
            to_scale: Index of the coarser scale
            
        Returns:
            Aggregated features at the coarser scale
        """
        # Use the cluster mapping to aggregate features
        # This is a simplified implementation - real aggregation would use
        # the actual graph structure and neighborhood information
        
        # Get the mapping from fine to coarse
        mapping = {}
        for i in range(from_scale, to_scale):
            if i == from_scale:
                mapping = self.cluster_mappings[i].copy()
            else:
                mapping = {k: self.cluster_mappings[i][v] for k, v in mapping.items()}
        
        # Group nodes by their cluster
        clusters = {}
        for node_idx, cluster_idx in mapping.items():
            if cluster_idx not in clusters:
                clusters[cluster_idx] = []
            clusters[cluster_idx].append(node_idx)
        
        # Aggregate features for each cluster
        aggregated_features = torch.zeros(
            (len(clusters), features.shape[1]), device=features.device
        )
        
        for i, (cluster_idx, node_indices) in enumerate(clusters.items()):
            # Mean pooling for now, could be more sophisticated
            cluster_features = features[node_indices].mean(dim=0)
            aggregated_features[i] = cluster_features
        
        return aggregated_features
    
    def _distribute_features(
        self, features: torch.Tensor, from_scale: int, to_scale: int
    ) -> torch.Tensor:
        """
        Distribute features from a coarser scale to a finer scale.
        
        Args:
            features: Node features at the coarser scale
            from_scale: Index of the coarser scale
            to_scale: Index of the finer scale
            
        Returns:
            Distributed features at the finer scale
        """
        # Use the cluster mapping to distribute features
        # This is a simplified implementation - real distribution would use
        # the actual graph structure and neighborhood information
        
        # Get the mapping from fine to coarse
        mapping = {}
        for i in range(to_scale, from_scale):
            if i == to_scale:
                mapping = self.cluster_mappings[i].copy()
            else:
                mapping = {k: self.cluster_mappings[i][v] for k, v in mapping.items()}
        
        # Invert the mapping (coarse to fine)
        inverted_mapping = {}
        for node_idx, cluster_idx in mapping.items():
            if cluster_idx not in inverted_mapping:
                inverted_mapping[cluster_idx] = []
            inverted_mapping[cluster_idx].append(node_idx)
        
        # Get the number of nodes in the finer scale
        num_fine_nodes = max(mapping.keys()) + 1
        
        # Distribute features to the finer scale
        distributed_features = torch.zeros(
            (num_fine_nodes, features.shape[1]), device=features.device
        )
        
        for cluster_idx, node_indices in inverted_mapping.items():
            # Broadcast the coarse features to all fine nodes in this cluster
            for node_idx in node_indices:
                distributed_features[node_idx] = features[cluster_idx]
        
        return distributed_features
    
    def _broadcast_features(
        self, features: torch.Tensor, from_scale: int, to_scale: int
    ) -> torch.Tensor:
        """
        Broadcast features from a coarser scale to a finer scale without aggregation.
        
        Args:
            features: Node features at the coarser scale
            from_scale: Index of the coarser scale
            to_scale: Index of the finer scale
            
        Returns:
            Broadcasted features at the finer scale
        """
        # Similar to _distribute_features but for broadcasting query vectors
        
        # Get the mapping from fine to coarse
        mapping = {}
        for i in range(to_scale, from_scale):
            if i == to_scale:
                mapping = self.cluster_mappings[i].copy()
            else:
                mapping = {k: self.cluster_mappings[i][v] for k, v in mapping.items()}
        
        # Invert the mapping (coarse to fine)
        inverted_mapping = {}
        for node_idx, cluster_idx in mapping.items():
            if cluster_idx not in inverted_mapping:
                inverted_mapping[cluster_idx] = []
            inverted_mapping[cluster_idx].append(node_idx)
        
        # Create a lookup for cluster indices
        cluster_indices = {cluster_idx: i for i, cluster_idx in enumerate(inverted_mapping.keys())}
        
        # Get the number of nodes in the finer scale
        num_fine_nodes = max(mapping.keys()) + 1
        
        # Broadcast features to the finer scale
        broadcasted_features = torch.zeros(
            (num_fine_nodes, features.shape[1]), device=features.device
        )
        
        for cluster_idx, node_indices in inverted_mapping.items():
            # Broadcast the coarse features to all fine nodes in this cluster
            cluster_features = features[cluster_indices[cluster_idx]]
            for node_idx in node_indices:
                broadcasted_features[node_idx] = cluster_features
        
        return broadcasted_features


class HMGNN(nn.Module):
    """
    Hierarchical Molecular Graph Neural Network.
    
    This model processes molecular graphs at multiple scales (atom, functional group, motif)
    with cross-scale information exchange for enhanced molecular property prediction.
    """
    
    def __init__(
        self,
        scale_dims: List[int],           # Node feature dimensions for each scale
        hidden_dim: int = 64,            # Hidden dimension
        n_blocks: int = 2,               # Number of GNN blocks per scale
        layers_per_block: int = 3,       # Number of layers per block
        edge_attr_dims: List[int] = None,# Edge feature dimensions for each scale
        jk_mode: str = 'attention',      # JK aggregation mode
        node_out_dim: int = 1,           # Node-level output dimension
        graph_out_dim: int = 1,          # Graph-level output dimension
        cross_scale_exchange: bool = True,# Whether to use cross-scale information exchange
        dropout: float = 0.2
    ):
        """
        Initialize the HMGNN model.
        
        Args:
            scale_dims: List of node feature dimensions for each scale
            hidden_dim: Hidden dimension
            n_blocks: Number of GNN blocks per scale
            layers_per_block: Number of layers per block
            edge_attr_dims: List of edge feature dimensions for each scale
            jk_mode: JK aggregation mode
            node_out_dim: Node-level output dimension
            graph_out_dim: Graph-level output dimension
            cross_scale_exchange: Whether to use cross-scale information exchange
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_scales = len(scale_dims)
        self.hidden_dim = hidden_dim
        self.cross_scale_exchange = cross_scale_exchange
        
        # Default edge attribute dimensions if not provided
        if edge_attr_dims is None:
            edge_attr_dims = [0] * self.num_scales
        
        # Verify we have the right number of edge dimensions
        assert len(edge_attr_dims) == self.num_scales, \
            f"Expected {self.num_scales} edge dimensions, got {len(edge_attr_dims)}"
        
        # Create GNN blocks for each scale
        self.scale_gnns = nn.ModuleList()
        self.scale_jk_aggregators = nn.ModuleList()
        
        for scale_idx in range(self.num_scales):
            # Create blocks for this scale
            scale_blocks = nn.ModuleList()
            block_dims = []
            
            # First block
            scale_blocks.append(
                DenseGNNBlock(
                    in_dim=scale_dims[scale_idx],
                    hidden_dim=hidden_dim,
                    n_layers=layers_per_block,
                    transition_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dims[scale_idx]
                )
            )
            block_dims.append(hidden_dim)
            
            # Additional blocks
            for _ in range(n_blocks - 1):
                scale_blocks.append(
                    DenseGNNBlock(
                        in_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        n_layers=layers_per_block,
                        transition_dim=hidden_dim,
                        edge_attr_dim=edge_attr_dims[scale_idx]
                    )
                )
                block_dims.append(hidden_dim)
            
            # Add blocks to module list
            self.scale_gnns.append(scale_blocks)
            
            # JK aggregator for this scale
            self.scale_jk_aggregators.append(
                JKAggregator(
                    block_dims=block_dims,
                    out_dim=hidden_dim,
                    mode=jk_mode
                )
            )
        
        # Cross-scale attention for information exchange
        if cross_scale_exchange:
            self.cross_scale_attention = CrossScaleAttention(
                scale_dims=[hidden_dim] * self.num_scales,
                hidden_dim=hidden_dim
            )
        
        # Node-level prediction heads for each scale
        self.node_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, node_out_dim)
            )
            for _ in range(self.num_scales)
        ])
        
        # Graph-level prediction heads for each scale
        self.graph_pools = nn.ModuleList([
            global_mean_pool for _ in range(self.num_scales)
        ])
        
        self.graph_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, graph_out_dim)
            )
            for _ in range(self.num_scales)
        ])
        
        # Final graph-level prediction head that combines all scales
        self.combined_graph_head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, graph_out_dim)
        )
    
    def forward(
        self,
        scale_data: List[Dict[str, Any]],
        cluster_mappings: Optional[List[Dict[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass of the HMGNN model.
        
        Args:
            scale_data: List of dictionaries containing data for each scale
                Each dictionary should have the following keys:
                - x: Node features
                - edge_index: Edge indices
                - edge_attr: Edge features (optional)
                - batch: Batch indices (optional)
            cluster_mappings: List of dictionaries mapping node indices from one scale to the next
                
        Returns:
            Dictionary with predictions at each scale and combined predictions
        """
        # Set cluster mappings for cross-scale attention
        if self.cross_scale_exchange and cluster_mappings is not None:
            self.cross_scale_attention.set_cluster_mappings(cluster_mappings)
        
        # Process each scale
        scale_block_outputs = []
        scale_final_features = []
        
        for scale_idx in range(self.num_scales):
            # Get data for this scale
            x = scale_data[scale_idx]['x']
            edge_index = scale_data[scale_idx]['edge_index']
            edge_attr = scale_data[scale_idx].get('edge_attr', None)
            
            # Apply GNN blocks
            block_outputs = []
            h = x
            
            for block in self.scale_gnns[scale_idx]:
                h = block(h, edge_index, edge_attr)
                block_outputs.append(h)
            
            # JK aggregation
            jk_features = self.scale_jk_aggregators[scale_idx](block_outputs)
            
            # Save outputs
            scale_block_outputs.append(block_outputs)
            scale_final_features.append(jk_features)
        
        # Cross-scale attention for information exchange
        if self.cross_scale_exchange and cluster_mappings is not None:
            scale_final_features = self.cross_scale_attention(scale_final_features)
        
        # Make predictions at each scale
        scale_node_preds = []
        scale_graph_preds = []
        scale_graph_embeds = []
        
        for scale_idx in range(self.num_scales):
            # Get batch indices for this scale
            batch = scale_data[scale_idx].get('batch', None)
            
            # Node-level prediction
            node_pred = self.node_heads[scale_idx](scale_final_features[scale_idx])
            scale_node_preds.append(node_pred)
            
            # Graph-level prediction
            if batch is None:
                # Single-graph scenario
                graph_embed = scale_final_features[scale_idx].mean(dim=0, keepdim=True)
            else:
                # Multi-graph scenario
                graph_embed = self.graph_pools[scale_idx](scale_final_features[scale_idx], batch)
            
            graph_pred = self.graph_heads[scale_idx](graph_embed)
            scale_graph_preds.append(graph_pred)
            scale_graph_embeds.append(graph_embed)
        
        # Combined graph-level prediction
        if len(scale_graph_embeds) > 1:
            combined_graph_embed = torch.cat(scale_graph_embeds, dim=1)
            combined_graph_pred = self.combined_graph_head(combined_graph_embed)
        else:
            combined_graph_pred = scale_graph_preds[0]
        
        # Build results dictionary
        results = {
            'node_pred': scale_node_preds[0],  # Atom-level predictions (default)
            'graph_pred': combined_graph_pred  # Combined graph-level predictions (default)
        }
        
        # Add scale-specific predictions
        for scale_idx in range(self.num_scales):
            scale_name = f"scale_{scale_idx}"
            results[f"{scale_name}_node_pred"] = scale_node_preds[scale_idx]
            results[f"{scale_name}_graph_pred"] = scale_graph_preds[scale_idx]
        
        return results


def create_hierarchical_mgnn(
    scale_dims: List[int],
    hidden_dim: int = 64,
    n_blocks: int = 2,
    layers_per_block: int = 3,
    edge_attr_dims: Optional[List[int]] = None,
    jk_mode: str = 'attention',
    node_out_dim: int = 1,
    graph_out_dim: int = 1,
    cross_scale_exchange: bool = True,
    dropout: float = 0.2
) -> HMGNN:
    """
    Create a Hierarchical Molecular Graph Neural Network.
    
    Args:
        scale_dims: List of node feature dimensions for each scale
        hidden_dim: Hidden dimension
        n_blocks: Number of GNN blocks per scale
        layers_per_block: Number of layers per block
        edge_attr_dims: List of edge feature dimensions for each scale
        jk_mode: JK aggregation mode
        node_out_dim: Node-level output dimension
        graph_out_dim: Graph-level output dimension
        cross_scale_exchange: Whether to use cross-scale information exchange
        dropout: Dropout rate
        
    Returns:
        HMGNN model
    """
    return HMGNN(
        scale_dims=scale_dims,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        layers_per_block=layers_per_block,
        edge_attr_dims=edge_attr_dims,
        jk_mode=jk_mode,
        node_out_dim=node_out_dim,
        graph_out_dim=graph_out_dim,
        cross_scale_exchange=cross_scale_exchange,
        dropout=dropout
    )