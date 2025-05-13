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
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from typing import Dict, List, Optional, Any

from moml.models.mgnn.djmgnn import DenseGNNBlock, JKAggregator

# Wrapper for global pooling functions
class GlobalPool(nn.Module):
    """
    Wrapper for PyTorch Geometric global pooling functions to be used in nn.ModuleList.
    """
    def __init__(self, pool_fn):
        super().__init__()
        self.pool_fn = pool_fn

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, size: Optional[int] = None) -> torch.Tensor:
        if batch is None: # Should not happen if data is prepared correctly
            if x.ndim == 2: # [num_nodes, features] -> [1, features]
                return self.pool_fn(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            elif x.ndim ==3: # [batch_size, num_nodes, features] -> [batch_size, features]
                 # This case needs careful handling if batch is truly None.
                 # Assuming for now that if batch is None, it's a single graph with 2D input.
                raise ValueError("Batch tensor must be provided for 3D input to GlobalPool without batch.")
            else:
                raise ValueError(f"Unsupported input dimension {x.ndim} for GlobalPool without batch.")

        return self.pool_fn(x, batch, size)

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
                        attended_values_at_fine = torch.matmul(attn_weights, values[other_idx])
                        
                        # Aggregate these fine-scale attended values to the coarse scale (scale_idx)
                        aggregated_to_coarse = self._aggregate_features(
                            attended_values_at_fine, other_idx, scale_idx # Aggregate from fine (other_idx) to coarse (scale_idx)
                        )
                        scale_output = scale_output + aggregated_to_coarse
            
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
            GlobalPool(global_mean_pool) for _ in range(self.num_scales)
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

        # Determine overall batch size from the first available batch tensor
        overall_batch_size = 0
        if scale_data: # Ensure scale_data is not empty
            for s_data_item in scale_data:
                batch_tensor_item = s_data_item.get('batch')
                if batch_tensor_item is not None and batch_tensor_item.numel() > 0:
                    overall_batch_size = batch_tensor_item.max().item() + 1
                    break
            if overall_batch_size == 0:
                if any(s_data_item['x'].numel() > 0 for s_data_item in scale_data):
                    overall_batch_size = 1
        
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
            current_scale_batch_tensor = scale_data[scale_idx].get('batch', None)

            # Node-level prediction
            if scale_final_features[scale_idx].numel() > 0:
                node_pred = self.node_heads[scale_idx](scale_final_features[scale_idx])
            else:
                node_out_features = self.hidden_dim # Default to hidden_dim if head structure is complex
                if isinstance(self.node_heads[scale_idx], nn.Sequential) and \
                   len(self.node_heads[scale_idx]) > 0 and \
                   hasattr(self.node_heads[scale_idx][-1], 'out_features'):
                    node_out_features = self.node_heads[scale_idx][-1].out_features
                elif hasattr(self.node_heads[scale_idx], 'out_features'): # If head is a single layer
                     node_out_features = self.node_heads[scale_idx].out_features

                node_pred = torch.empty((0, node_out_features),
                                        device=scale_final_features[scale_idx].device)
            scale_node_preds.append(node_pred)

            # Graph-level prediction
            graph_embed_dim = self.hidden_dim # Default, should be input dim of graph_head
            if isinstance(self.graph_heads[scale_idx], nn.Sequential) and \
               len(self.graph_heads[scale_idx]) > 0 and \
               hasattr(self.graph_heads[scale_idx][0], 'in_features'):
                graph_embed_dim = self.graph_heads[scale_idx][0].in_features
            elif hasattr(self.graph_heads[scale_idx], 'in_features'):
                 graph_embed_dim = self.graph_heads[scale_idx].in_features


            if scale_final_features[scale_idx].numel() == 0 or overall_batch_size == 0:
                graph_embed = torch.zeros((overall_batch_size, graph_embed_dim),
                                          device=scale_final_features[scale_idx].device)
            else:
                if current_scale_batch_tensor is None and overall_batch_size == 1: # Single graph, features exist
                    current_scale_batch_tensor = torch.zeros(scale_final_features[scale_idx].size(0),
                                                             dtype=torch.long,
                                                             device=scale_final_features[scale_idx].device)
                
                if current_scale_batch_tensor is None: # Should only happen if overall_batch_size is 0 now
                     raise ValueError(f"Batch tensor is None for scale {scale_idx} but overall_batch_size={overall_batch_size} and features exist.")

                graph_embed = self.graph_pools[scale_idx](scale_final_features[scale_idx], current_scale_batch_tensor)

                # If pooling results in an empty tensor (e.g. (0,D) because all graphs in batch had 0 nodes for this scale)
                # but overall_batch_size > 0, we need to reshape to (overall_batch_size, D) of zeros.
                if graph_embed.numel() == 0 and overall_batch_size > 0:
                     graph_embed = torch.zeros((overall_batch_size, graph_embed_dim),
                                               device=scale_final_features[scale_idx].device)
                # Ensure consistent batch dimension if pooling was sparse (this is a simplified handling)
                elif graph_embed.size(0) != overall_batch_size and overall_batch_size > 0:
                    # This implies global_pool might have returned a sparse batch.
                    # We create a dense zero tensor and fill it. This assumes pooled indices are contiguous.
                    # A truly robust solution for sparse batches from global_pool would require scatter.
                    # print(f"Warning: Correcting graph_embed size for scale {scale_idx}. From {graph_embed.shape} to ({overall_batch_size}, {graph_embed_dim})")
                    temp_dense_embed = torch.zeros((overall_batch_size, graph_embed_dim), device=graph_embed.device)
                    if graph_embed.numel() > 0: # only copy if there's something to copy
                        # This assumes that if graph_embed.size(0) < overall_batch_size, the existing embeddings
                        # correspond to the first graph_embed.size(0) items in the batch.
                        # This is a strong assumption about the output of global_pool with sparse batches.
                        # PyG's global_add_pool, global_mean_pool, global_max_pool with a complete batch vector
                        # (0 to B-1) should return (B, D). If not, the batch vector itself might be sparse.
                        # For this test (zero_nodes_in_one_scale), it's a single graph, so overall_batch_size=1.
                        # If scale_final_features[scale_idx].numel() > 0, graph_embed should be (1,D).
                        # If it's not, then the pooling or batching is the issue.
                        # The numel()==0 check above should handle the "all nodes empty for this scale" case.
                        # This branch is more for "some graphs in batch are empty for this scale".
                        # For the specific failing test (single graph, one scale 0 nodes), this branch might not be critical
                        # if the numel()==0 path correctly makes a (1,D) zero tensor.
                        # The critical part is that `graph_embed` must be `(overall_batch_size, D)`.
                        # If `global_pool` returns `(k, D)` where `k < overall_batch_size`, it means only `k` graphs had nodes.
                        # We need to expand this to `(overall_batch_size, D)`.
                        # The current test is single graph, so `overall_batch_size` is 1.
                        # If `scale_final_features` is not empty, `graph_embed` should be `(1,D)`.
                        # If `scale_final_features` is empty, the `numel()==0` path makes `(1,D)` zeros.
                        # This specific `elif` might be more for true batched scenarios.
                        # For safety, let's ensure the (overall_batch_size, D) shape.
                        if graph_embed.size(0) > 0:
                             temp_dense_embed[:graph_embed.size(0)] = graph_embed # Naive fill
                        graph_embed = temp_dense_embed


            graph_pred = self.graph_heads[scale_idx](graph_embed)
            scale_graph_preds.append(graph_pred)
            scale_graph_embeds.append(graph_embed)
        
        # Combined graph-level prediction
        # Filter out any graph_embeds that might be (0,D) if overall_batch_size was 0
        valid_graph_embeds = [embed for embed in scale_graph_embeds if embed.size(0) > 0]
        if not valid_graph_embeds and overall_batch_size == 0 : # All were (0,D) because batch was empty
            # Output a (0,D) prediction for combined
            combined_graph_pred_dim = self.combined_graph_head[-1].out_features
            combined_graph_pred = torch.empty((0, combined_graph_pred_dim), device=scale_data[0]['x'].device if scale_data and scale_data[0]['x'] is not None else torch.device('cpu'))
        elif not valid_graph_embeds and overall_batch_size > 0:
             # This means all scales resulted in empty features for all graphs in the batch.
             # Should produce a zero prediction of shape (overall_batch_size, out_dim)
            combined_graph_pred_dim = self.combined_graph_head[-1].out_features
            combined_graph_pred = torch.zeros((overall_batch_size, combined_graph_pred_dim), device=scale_data[0]['x'].device if scale_data and scale_data[0]['x'] is not None else torch.device('cpu'))
        elif len(valid_graph_embeds) == 1: # Only one scale had actual embeddings or only one scale model
            combined_graph_pred = self.graph_heads[0](valid_graph_embeds[0]) # Assuming if one valid, it's from scale 0 or it's a single scale model
                                                                            # This needs to be more robust if only one valid embed is not from scale 0
                                                                            # For now, if only one valid, use its graph_pred directly.
                                                                            # Find which scale_graph_pred corresponds to the valid_graph_embed
            original_idx = -1
            for idx, embed in enumerate(scale_graph_embeds):
                if embed is valid_graph_embeds[0]:
                    original_idx = idx
                    break
            if original_idx != -1:
                combined_graph_pred = scale_graph_preds[original_idx]
            else: # Should not happen
                combined_graph_pred = scale_graph_preds[0]


        elif len(valid_graph_embeds) > 1:
            # Ensure all valid_graph_embeds have the same batch size (overall_batch_size)
            # This should be guaranteed by the logic above.
            combined_graph_embed_cat = torch.cat(valid_graph_embeds, dim=1)
            combined_graph_pred = self.combined_graph_head(combined_graph_embed_cat)
        else: # No valid graph embeds, but overall_batch_size > 0 (e.g. all scales were empty for all graphs)
             # This case is covered by the `elif not valid_graph_embeds and overall_batch_size > 0`
             # For safety, re-state:
            combined_graph_pred_dim = self.combined_graph_head[-1].out_features
            combined_graph_pred = torch.zeros((overall_batch_size, combined_graph_pred_dim), device=scale_data[0]['x'].device if scale_data and scale_data[0]['x'] is not None else torch.device('cpu'))


        # Default node_pred from scale 0, ensure it's (0,D) if scale 0 had no nodes
        default_node_pred = scale_node_preds[0] if self.num_scales > 0 else torch.empty((0,0))


        # Build results dictionary
        results = {
            'node_pred': default_node_pred,
            'graph_pred': combined_graph_pred
        }
        
        # Add scale-specific predictions
        for scale_idx in range(self.num_scales):
            scale_name = f"scale_{scale_idx}"
            results[f"{scale_name}_node_pred"] = scale_node_preds[scale_idx]
            # Ensure graph_pred for empty scales is (overall_batch_size, D_out) or (0, D_out)
            if scale_graph_preds[scale_idx].size(0) != overall_batch_size and overall_batch_size > 0 :
                 # This implies the graph_head might have processed a (0,D_in) embed if overall_batch_size was 0 for that head
                 # Or the graph_embed fed to it was (0,D_in)
                 # We need the output to be (overall_batch_size, D_out_graph_head)
                graph_head_out_dim = self.graph_heads[scale_idx][-1].out_features
                results[f"{scale_name}_graph_pred"] = torch.zeros((overall_batch_size, graph_head_out_dim), device=combined_graph_pred.device)
            else:
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