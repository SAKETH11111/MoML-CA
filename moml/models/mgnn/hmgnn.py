
"""
moml/models/mgnn/hmgnn.py

Hierarchical Molecular Graph Neural Network (HMGNN)

This module implements a hierarchical molecular graph neural network for 
multi-scale molecular representation learning. It provides cross-scale 
attention mechanisms and multi-resolution processing for comprehensive 
molecular analysis.

The implementation focuses on multi-scale/cross-scale machinery while 
reusing core building blocks (DenseGNNBlock, JKAggregator) from djmgnn.py.

Main Components:
    - CrossScaleAttentionMH: Multi-head attention across scales
    - EdgeNNConv: Edge-conditioned convolution for bipartite graphs
    - HMGNN: Complete hierarchical molecular GNN model
    - Helper functions: Scale aggregation and broadcasting utilities
"""

from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_scatter import scatter_add

from moml.models.mgnn.djmgnn import DenseGNNBlock, JKAggregator

logger = logging.getLogger(__name__)

# Helper functions for cross-scale feature processing
def aggregate_fine_to_coarse(
    feat: torch.Tensor, 
    fine2coarse: torch.Tensor, 
    coarse_count: torch.Tensor
) -> torch.Tensor:
    """
    Mean-pool fine-scale features to coarse clusters.
    
    Args:
        feat: Fine-scale node features [num_fine_nodes, feat_dim]
        fine2coarse: Mapping from fine to coarse nodes [num_fine_nodes]
        coarse_count: Number of fine nodes per coarse node [num_coarse_nodes]
        
    Returns:
        torch.Tensor: Aggregated coarse features [num_coarse_nodes, feat_dim]
    """
    # Handle dimension mismatch between feat and fine2coarse
    if feat.shape[0] != fine2coarse.shape[0]:
        # This can happen with batched graphs where the mapping is for single graphs
        # but features are for batched graphs
        if feat.shape[0] > fine2coarse.shape[0]:
            # Repeat the mapping for each graph in the batch
            batch_size = feat.shape[0] // fine2coarse.shape[0]
            if feat.shape[0] % fine2coarse.shape[0] == 0:
                # Clean multiple - repeat the mapping
                offset = coarse_count.shape[0]
                repeated_mappings = []
                for i in range(batch_size):
                    repeated_mappings.append(fine2coarse + i * offset)
                fine2coarse = torch.cat(repeated_mappings)
                
                # Also repeat and concatenate coarse_count
                coarse_count = torch.cat([coarse_count] * batch_size)
            else:
                # Truncate feat to match fine2coarse
                feat = feat[:fine2coarse.shape[0]]
        else:
            # Truncate fine2coarse to match feat
            fine2coarse = fine2coarse[:feat.shape[0]]
    
    # Ensure scatter_add produces the right size by specifying dim_size
    max_index = fine2coarse.max().item() if fine2coarse.numel() > 0 else -1
    expected_size = max(max_index + 1, coarse_count.shape[0])
    
    summed = scatter_add(feat, fine2coarse, dim=0, dim_size=expected_size)
    
    # Ensure coarse_count matches the summed tensor size
    if summed.shape[0] != coarse_count.shape[0]:
        # Pad or truncate coarse_count to match summed tensor size
        if summed.shape[0] > coarse_count.shape[0]:
            # Pad with ones (meaning single node per cluster for missing clusters)
            pad_size = summed.shape[0] - coarse_count.shape[0]
            coarse_count = torch.cat([coarse_count, torch.ones(pad_size, device=coarse_count.device, dtype=coarse_count.dtype)])
        else:
            # Truncate 
            coarse_count = coarse_count[:summed.shape[0]]
    
    # Ensure coarse_count has the right shape for broadcasting
    # summed has shape [num_coarse_nodes, ...], coarse_count has shape [num_coarse_nodes]
    # We need to expand coarse_count to match the shape of summed
    coarse_count_expanded = coarse_count.view(coarse_count.shape[0], *[1] * (summed.ndim - 1))
    
    return summed / coarse_count_expanded.clamp(min=1)


def broadcast_coarse_to_fine(
    coarse_feat: torch.Tensor, 
    fine2coarse: torch.Tensor
) -> torch.Tensor:
    """
    Broadcast coarse features to their member fine nodes.
    
    Args:
        coarse_feat: Coarse-scale node features [num_coarse_nodes, feat_dim]
        fine2coarse: Mapping from fine to coarse nodes [num_fine_nodes]
        
    Returns:
        torch.Tensor: Broadcasted fine features [num_fine_nodes, feat_dim]
    """
    # Ensure indices are within bounds
    if len(coarse_feat) == 0:
        return torch.empty(0, coarse_feat.shape[1], dtype=coarse_feat.dtype, device=coarse_feat.device)
    
    # Clamp indices to valid range
    fine2coarse_clamped = torch.clamp(fine2coarse, 0, len(coarse_feat) - 1)
    return coarse_feat[fine2coarse_clamped]


# Core layer classes
class EdgeNNConv(nn.Module):
    """
    Edge-conditioned convolution for bipartite cross-scale graphs.
    
    Implements message passing between different scales using edge features
    to modulate the transformation of source node features.
    """
    
    def __init__(self, in_src: int, in_tgt: int, out_dim: int, edge_dim: int) -> None:
        """
        Initialize EdgeNNConv layer.
        
        Args:
            in_src: Source node feature dimension
            in_tgt: Target node feature dimension (unused in current implementation)
            out_dim: Output feature dimension
            edge_dim: Edge feature dimension
        """
        super().__init__()
        self.out_dim = out_dim
        self.nn = nn.Sequential(
            nn.Linear(edge_dim, in_src * out_dim), 
            nn.ReLU()
        )

    def forward(
        self, 
        x_src: torch.Tensor, 
        x_tgt: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for edge-conditioned convolution.
        
        Args:
            x_src: Source node features [num_src_nodes, in_src]
            x_tgt: Target node features [num_tgt_nodes, in_tgt]
            edge_index: Edge connectivity [2, num_edges] 
                       (row 0 = source indices, row 1 = target indices)
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            torch.Tensor: Aggregated messages for target nodes [num_tgt_nodes, out_dim]
        """
        w = self.nn(edge_attr).view(-1, self.out_dim, x_src.size(1))
        msg = torch.bmm(w, x_src[edge_index[0]].unsqueeze(-1)).squeeze(-1)
        return scatter_add(msg, edge_index[1], dim=0, dim_size=x_tgt.size(0))


class CrossScaleAttentionMH(nn.Module):
    """
    Multi-head cross-scale attention for hierarchical graph networks.
    
    Implements multi-head attention across multiple scales with optional
    edge-conditioned message passing between fine and coarse scales.
    Scale 0 is the finest scale.
    """

    def __init__(
        self, 
        n_scales: int, 
        hidden_dim: int, 
        n_heads: int = 4, 
        edge_dim: int = 0
    ) -> None:
        """
        Initialize cross-scale attention module.
        
        Args:
            n_scales: Number of scales in the hierarchy
            hidden_dim: Hidden feature dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            edge_dim: Edge feature dimension for cross-scale connections
            
        Raises:
            AssertionError: If hidden_dim is not divisible by n_heads
        """
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.S, self.h, self.d_k = n_scales, n_heads, hidden_dim // n_heads
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.d_k)), requires_grad=True)
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        self.q_proj = self._make_proj()
        self.k_proj = self._make_proj()
        self.v_proj = self._make_proj()
        self.out_proj = self._make_proj()

        self.edge_msg = (
            nn.ModuleList([EdgeNNConv(hidden_dim, hidden_dim, hidden_dim, edge_dim) for _ in range(n_scales - 1)])
            if edge_dim
            else None
        )

    # Helper methods
    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into multiple attention heads."""
        return x.view(x.size(0), self.h, self.d_k)

    def _join(self, x: torch.Tensor) -> torch.Tensor:
        """Join multiple attention heads back into single tensor."""
        return x.view(x.size(0), self.h * self.d_k)

    def _make_proj(self) -> nn.ModuleList:
        """
        Create linear projections for each scale.
        
        Returns:
            nn.ModuleList: List of linear layers (hidden_dim → hidden_dim) for each scale
        """
        return nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_scales)])

    def forward(
        self,
        feats: List[torch.Tensor],  # per-scale node embeddings
        maps: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]],  # Made maps optional
        edge_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        fine2coarse, coarse_count = (None, None)
        can_map = False
        if maps is not None:
            if isinstance(maps, tuple) and len(maps) == 2:
                fine2coarse, coarse_count = maps
                if fine2coarse is None:
                    fine2coarse = []
                if coarse_count is None:
                    coarse_count = []
                can_map = True
            else:
                logger.warning(
                    "CrossScaleAttentionMH: 'maps' argument provided but not in expected format (Tuple[List[Tensor], List[Tensor]]). Proceeding without mapping."
                )
                fine2coarse, coarse_count = [], []
        else:
            fine2coarse, coarse_count = [], []

        # Apply edge-conditioned messages between scales
        if self.edge_msg and edge_pairs is not None:
            for i in range(self.S - 1):
                ei, ea = edge_pairs[i]
                msg = self.edge_msg[i](feats[i], feats[i + 1], ei, ea)
                feats[i + 1] = feats[i + 1] + msg

        updated = []
        for t in range(self.S):  # target scale
            q = self._split(self.q_proj[t](feats[t]))
            agg = torch.zeros_like(q)

            for s in range(self.S):  # source scale
                # Only allow self-attention if no cross-scale mapping available
                if not can_map and s != t:
                    continue

                k = self._split(self.k_proj[s](feats[s]))
                v = self._split(self.v_proj[s](feats[s]))

                if s != t and can_map:
                    if s < t:  # fine → coarse: aggregate
                        if (
                            s < len(fine2coarse)
                            and s < len(coarse_count)
                            and fine2coarse[s] is not None
                            and coarse_count[s] is not None
                        ):
                            k_agg = aggregate_fine_to_coarse(k, fine2coarse[s], coarse_count[s])
                            v_agg = aggregate_fine_to_coarse(v, fine2coarse[s], coarse_count[s])
                            if k_agg.shape[0] == q.shape[0]:
                                k = k_agg
                                v = v_agg
                            else:
                                logger.warning(
                                    f"CrossScaleAttn: Aggregation shape mismatch for s={s}, t={t}. k_agg: {k_agg.shape}, q: {q.shape}. Using original k,v."
                                )
                        else:
                            logger.debug(
                                f"CrossScaleAttn: Not enough mapping info for s={s} < t={t}. Using original k,v."
                            )
                    else:
                        if t < len(fine2coarse) and fine2coarse[t] is not None:
                            k_broad = broadcast_coarse_to_fine(k, fine2coarse[t])
                            v_broad = broadcast_coarse_to_fine(v, fine2coarse[t])
                            if k_broad.shape[0] == q.shape[0]:
                                k = k_broad
                                v = v_broad
                            else:
                                logger.warning(
                                    f"CrossScaleAttn: Broadcast shape mismatch for s={s}, t={t}. k_broad: {k_broad.shape}, q: {q.shape}. Using original k,v."
                                )
                        else:
                            logger.debug(
                                f"CrossScaleAttn: Not enough mapping info for s={s} > t={t}. Using original k,v."
                            )

                # Check if q and k have compatible shapes
                if len(q.shape) < 3 or len(k.shape) < 3 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
                    logger.warning(
                        f"CrossScaleAttn: Incompatible shapes for attention: q={q.shape}, k={k.shape}. Skipping attention for s={s}, t={t}."
                    )
                    continue
                
                scores = (q * k).sum(-1) / self.scale
                w = scores.softmax(0).unsqueeze(-1)
                agg = agg + w * v

            updated.append(feats[t] + self.out_proj[t](self._join(agg)))
        return updated


class HMGNN(nn.Module):
    """
    Hierarchical Molecular Graph Neural Network.
    
    Multi-resolution GNN with cross-scale attention and uncertainty-weighted loss
    for comprehensive molecular representation learning across multiple scales.
    
    The model processes molecular data at different resolutions simultaneously
    and uses cross-scale attention to exchange information between scales.
    """

    def __init__(
        self,
        scale_dims: List[int],
        hidden_dim: int = 64,
        env_dim: int = 0,
        env_mlp: bool = False,
        n_blocks: int = 2,
        layers_per_block: int = 3,
        edge_attr_dims: Optional[List[int]] = None,
        jk_mode: str = "attention",
        node_out_dim: int = 1,
        graph_out_dim: int = 1,
        cross_scale_exchange: bool = True,
        dropout: float = 0.2,
        n_heads_cs: int = 4,
        edge_dim_cs: int = 0,
        pool_type: str = "mean",
    ) -> None:
        """
        Initialize Hierarchical Molecular GNN.
        
        Args:
            scale_dims: Input feature dimensions for each scale
            hidden_dim: Hidden layer dimension
            env_dim: Environmental feature dimension
            env_mlp: Whether to use MLP for environment features
            n_blocks: Number of GNN blocks per scale
            layers_per_block: Number of layers per GNN block
            edge_attr_dims: Edge attribute dimensions for each scale
            jk_mode: Jumping knowledge aggregation mode
            node_out_dim: Output dimension for node predictions
            graph_out_dim: Output dimension for graph predictions
            cross_scale_exchange: Whether to use cross-scale attention
            dropout: Dropout rate
            n_heads_cs: Number of attention heads for cross-scale attention
            edge_dim_cs: Edge dimension for cross-scale attention
            pool_type: Graph pooling type ('mean', 'add', 'max')
        """
        super().__init__()
        self.env_dim = env_dim
        self.S = len(scale_dims)
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.node_out_dim = node_out_dim
        self.graph_out_dim = graph_out_dim
        self.env_dim = env_dim
        edge_attr_dims = edge_attr_dims or [0] * self.S

        # backbone per scale
        self.scale_gnns, self.scale_jk = nn.ModuleList(), nn.ModuleList()
        for d_in, e_dim in zip(scale_dims, edge_attr_dims):
            blocks, dims = nn.ModuleList(), []
            # first block
            blocks.append(DenseGNNBlock(d_in, hidden_dim, layers_per_block, hidden_dim, e_dim))
            dims.append(hidden_dim)
            # subsequent blocks
            for _ in range(n_blocks - 1):
                blocks.append(DenseGNNBlock(hidden_dim, hidden_dim, layers_per_block, hidden_dim, e_dim))
                dims.append(hidden_dim)
            self.scale_gnns.append(blocks)
            self.scale_jk.append(JKAggregator(dims, hidden_dim, mode=jk_mode))

        # cross-scale module
        self.use_cs = cross_scale_exchange
        if cross_scale_exchange:
            self.cross_scale = CrossScaleAttentionMH(self.S, hidden_dim, n_heads_cs, edge_dim_cs)

        # env projection
        env_in = env_dim if not env_mlp else hidden_dim
        if env_dim and env_mlp:
            self.env_proj = nn.Sequential(nn.Linear(env_dim, hidden_dim), nn.SiLU())
        else:
            self.env_proj = None

        fused_dim = hidden_dim + env_in

        # heads
        self.node_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, node_out_dim),
                )
                for _ in range(self.S)
            ]
        )
        pool_dict = {"mean": global_mean_pool, "add": global_add_pool, "max": global_max_pool}
        self.graph_pool = pool_dict.get(pool_type, global_mean_pool)
        self.graph_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, graph_out_dim),
                )
                for _ in range(self.S)
            ]
        )
        self.combined_graph_head = nn.Sequential(
            nn.Linear(fused_dim * self.S, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, graph_out_dim),
        )

        self.log_sigma_node = nn.Parameter(torch.zeros(1))
        self.log_sigma_graph = nn.Parameter(torch.zeros(1))

    @property
    def gnn_blocks(self):
        """Alias for scale_gnns to maintain backward compatibility."""
        return self.scale_gnns

    @property
    def cross_scale_attention(self):
        """Alias for cross_scale to maintain backward compatibility."""
        return self.cross_scale

    def forward(
        self,
        scale_data: List[Dict[str, Any]],
        maps: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
        edge_pairs_cs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        env_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        scale_data[i] = {'x', 'edge_index', 'edge_attr'?, 'batch'?, 'mask'?}
        maps          = (fine2coarse, coarse_count)   – each list length S-1
        edge_pairs_cs = list length S-1 of (edge_index, edge_attr) for fine→coarse
        """
        scale_feats, graph_embeds, node_preds, graph_preds = [], [], [], []

        for i in range(self.S):
            x = scale_data[i]["x"]
            ei = scale_data[i]["edge_index"]
            ea = scale_data[i].get("edge_attr", None)

            outs, h = [], x
            for block in self.scale_gnns[i].children():
                h = block(h, ei, ea)
                outs.append(h)
            z = self.scale_jk[i](outs)
            scale_feats.append(z)

        # cross-scale attention
        if self.use_cs and maps is not None:
            scale_feats = self.cross_scale(scale_feats, maps, edge_pairs_cs)

        output_dict = {}
        for i in range(self.S):
            batch_vec = scale_data[i].get("batch", None)
            current_scale_node_feats = scale_feats[i]

            current_node_pred = self.node_heads[i](current_scale_node_feats)
            node_preds.append(current_node_pred)
            output_dict[f"scale_{i}_node_pred"] = current_node_pred

            if current_scale_node_feats.size(0) == 0:
                num_graphs_in_batch = 1
                if batch_vec is not None and batch_vec.numel() > 0:
                    num_graphs_in_batch = batch_vec.max().item() + 1

                g_emb = torch.zeros((num_graphs_in_batch, self.hidden_dim), device=current_scale_node_feats.device)
                current_graph_pred = torch.zeros(
                    (num_graphs_in_batch, self.graph_out_dim), device=current_scale_node_feats.device
                )
            else:
                g_emb = (
                    current_scale_node_feats.mean(0, keepdim=True)
                    if batch_vec is None
                    else self.graph_pool(current_scale_node_feats, batch_vec)
                )
                if self.env_dim:
                    if env_vec is None:
                        env_vec = torch.zeros(g_emb.size(0), self.env_dim, device=g_emb.device)
                    env_emb = self.env_proj(env_vec) if self.env_proj else env_vec
                    g_emb = torch.cat([g_emb, env_emb], dim=1)
                current_graph_pred = self.graph_heads[i](g_emb)
            graph_embeds.append(g_emb)
            graph_preds.append(current_graph_pred)
            output_dict[f"scale_{i}_graph_pred"] = current_graph_pred

        if not graph_embeds:
            combined_graph_pred = torch.empty(0, device=x.device if "x" in scale_data[0] else "cpu")
        else:
            try:
                concatenated_graph_embeds = torch.cat(graph_embeds, dim=1)
                combined_graph_pred = self.combined_graph_head(concatenated_graph_embeds)
            except RuntimeError as e:
                logger.error(f"HMGNN: Error during torch.cat(graph_embeds)")
                logger.error(f"Shapes of graph_embeds: {[ge.shape for ge in graph_embeds]}")
                bs = int(graph_embeds[0].size(0)) if graph_embeds and graph_embeds[0].numel() > 0 else 1
                last_layer = self.combined_graph_head[-1]
                if hasattr(last_layer, 'out_features') and isinstance(last_layer.out_features, int):
                    out_dim_combined_graph_head = last_layer.out_features
                else:
                    out_dim_combined_graph_head = self.graph_out_dim
                combined_graph_pred = torch.zeros(
                    bs, out_dim_combined_graph_head,
                    device=graph_embeds[0].device if graph_embeds else torch.device("cpu")
                )

        output_dict["node_pred"] = node_preds[0] if node_preds else None
        output_dict["graph_pred"] = combined_graph_pred
        return output_dict

    def compute_losses(
        self, preds: Dict[str, Any], targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        targets = {'node': (y, mask) , 'graph': (y, mask)}
        """
        y_n, m_n = targets["node"]
        y_g, m_g = targets["graph"]

        node_loss = F.mse_loss(preds["node_pred"][m_n], y_n, reduction="mean")
        graph_loss = F.mse_loss(preds["graph_pred"][m_g], y_g, reduction="mean")

        total = (node_loss / torch.exp(self.log_sigma_node) + self.log_sigma_node) + (
            graph_loss / torch.exp(self.log_sigma_graph) + self.log_sigma_graph
        )
        return total, {"node": node_loss.item(), "graph": graph_loss.item()}


# Factory function
def create_hierarchical_mgnn(config: Dict[str, Any]) -> HMGNN:
    """
    Create a hierarchical MGNN model from a configuration dictionary.
    
    Args:
        config: Configuration dictionary containing model parameters:
            - scale_dims: List of input dimensions for each scale
            - hidden_dim: Hidden dimension size
            - edge_attr_dims: Optional list of edge attribute dimensions for each scale
            - node_out_dim: Output dimension for node predictions
            - graph_out_dim: Output dimension for graph predictions
            - n_blocks: Number of GNN blocks per scale
            - layers_per_block: Number of layers per block
            - jk_mode: Jumping knowledge mode
            - cross_scale_exchange: Whether to use cross-scale attention
            - dropout: Dropout rate
            - n_heads_cs: Number of attention heads for cross-scale attention
            - edge_dim_cs: Edge dimension for cross-scale attention
            - pool_type: Pooling type for graph-level predictions
    
    Returns:
        HMGNN: Configured hierarchical molecular GNN instance
    """
    edge_attr_dims = config.get("edge_attr_dims")
    if edge_attr_dims is None:
        edge_attr_dims = [0] * len(config["scale_dims"])
    
    return HMGNN(
        scale_dims=config["scale_dims"],
        hidden_dim=config["hidden_dim"],
        env_dim=0,
        env_mlp=False,
        n_blocks=config["n_blocks"],
        layers_per_block=config["layers_per_block"],
        edge_attr_dims=edge_attr_dims,
        jk_mode=config["jk_mode"],
        node_out_dim=config["node_out_dim"],
        graph_out_dim=config["graph_out_dim"],
        cross_scale_exchange=config["cross_scale_exchange"],
        dropout=config["dropout"],
        n_heads_cs=config["n_heads_cs"],
        edge_dim_cs=config["edge_dim_cs"],
        pool_type=config["pool_type"],
    )
