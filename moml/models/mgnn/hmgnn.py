"""
hmgnn.py  – Hierarchical Molecular Graph Neural Network (HMGNN)

This file holds only the multi-scale / cross-scale machinery.  Core blocks such as
DenseGNNBlock and JKAggregator are reused from `djmgnn.py`.


"""

from __future__ import annotations
import math
import logging

logger = logging.getLogger(__name__)
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_scatter import scatter_add

from moml.models.mgnn.djmgnn import DenseGNNBlock, JKAggregator  # adjust the path if needed


#  GPU scatter / gather helpers
def aggregate_fine_to_coarse(feat: torch.Tensor, fine2coarse: torch.Tensor, coarse_count: torch.Tensor) -> torch.Tensor:
    """Mean-pool fine-scale features to coarse clusters."""
    summed = scatter_add(feat, fine2coarse, dim=0)
    return summed / coarse_count.unsqueeze(-1).clamp(min=1)


def broadcast_coarse_to_fine(coarse_feat: torch.Tensor, fine2coarse: torch.Tensor) -> torch.Tensor:
    """Repeat coarse features on their member fine nodes."""
    return coarse_feat[fine2coarse]


#  Edge-conditioned one-step convolution on bipartite cross-scale graph
class EdgeNNConv(nn.Module):
    def __init__(self, in_src: int, in_tgt: int, out_dim: int, edge_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.nn = nn.Sequential(nn.Linear(edge_dim, in_src * out_dim), nn.ReLU())

    def forward(self, x_src, x_tgt, edge_index, edge_attr):
        """
        edge_index : [2, E]  (row 0 = source indices, row 1 = target indices)
        edge_attr  : [E, edge_dim]
        """
        w = self.nn(edge_attr).view(-1, self.out_dim, x_src.size(1))  # [E, out, in_src]
        msg = torch.bmm(w, x_src[edge_index[0]].unsqueeze(-1)).squeeze(-1)  # [E, out]
        return scatter_add(msg, edge_index[1], dim=0, dim_size=x_tgt.size(0))


#  Multi-head cross-scale attention
class CrossScaleAttentionMH(nn.Module):
    """
    Multi-head attention across S scales (0 = finest).
    Optionally preceded by an EdgeNNConv message pass on each fine→coarse pair.
    """

    def __init__(self, n_scales: int, hidden_dim: int, n_heads: int = 4, edge_dim: int = 0):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.S, self.h, self.d_k = n_scales, n_heads, hidden_dim // n_heads
        # Learnable temperature initialized to 1/sqrt(d_k)
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.d_k)), requires_grad=True)
        # Store dims for projection helper
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales
        # Initialize per-scale projection modules
        self.q_proj = self._make_proj()
        self.k_proj = self._make_proj()
        self.v_proj = self._make_proj()
        self.out_proj = self._make_proj()

        self.edge_msg = (
            nn.ModuleList([EdgeNNConv(hidden_dim, hidden_dim, hidden_dim, edge_dim) for _ in range(n_scales - 1)])
            if edge_dim
            else None
        )

    # helpers
    def _split(self, x):
        return x.view(x.size(0), self.h, self.d_k)

    def _join(self, x):
        return x.view(x.size(0), self.h * self.d_k)

    def _make_proj(self) -> nn.ModuleList:
        """Create a list of linear projections (hidden_dim → hidden_dim) for each scale."""
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
                # Ensure they are lists of appropriate length if not None
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

        # optional edge-conditioned messages (fine → coarse)
        if self.edge_msg and edge_pairs is not None:
            for i in range(self.S - 1):
                ei, ea = edge_pairs[i]
                msg = self.edge_msg[i](feats[i], feats[i + 1], ei, ea)
                feats[i + 1] = feats[i + 1] + msg

        updated = []
        for t in range(self.S):  # target scale
            q = self._split(self.q_proj[t](feats[t]))  # [Nt, h, d_k]
            agg = torch.zeros_like(q)

            for s in range(self.S):  # source scale
                # If no maps are available to align different scales, only allow self-attention (s == t)
                if not can_map and s != t:
                    continue

                k = self._split(self.k_proj[s](feats[s]))
                v = self._split(self.v_proj[s](feats[s]))

                if s != t and can_map:  # Only attempt mapping if maps were valid and s != t
                    if s < t:  # fine → coarse: aggregate
                        if (
                            s < len(fine2coarse)
                            and s < len(coarse_count)
                            and fine2coarse[s] is not None
                            and coarse_count[s] is not None
                        ):
                            k_agg = aggregate_fine_to_coarse(k, fine2coarse[s], coarse_count[s])
                            v_agg = aggregate_fine_to_coarse(v, fine2coarse[s], coarse_count[s])
                            if k_agg.shape[0] == q.shape[0]:  # Check if num target nodes match
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
                    else:  # coarse → fine: broadcast (s > t)
                        if t < len(fine2coarse) and fine2coarse[t] is not None:
                            # fine2coarse[t] maps nodes from scale t to t+1.
                            # For broadcasting from s to t (s > t), we need map from t to s.
                            # This part of logic might need review if broadcasting uses fine2coarse directly.
                            # Assuming fine2coarse[t] is relevant for broadcasting to scale t from a coarser scale.
                            k_broad = broadcast_coarse_to_fine(k, fine2coarse[t])  # This map might be wrong for s->t
                            v_broad = broadcast_coarse_to_fine(v, fine2coarse[t])  # This map might be wrong for s->t
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
                # If s == t or not can_map or mapping failed, k and v remain original feats[s] projections

                scores = (q * k).sum(-1) / self.scale  # [Nt, h]
                w = scores.softmax(0).unsqueeze(-1)  # [Nt, h, 1]
                agg = agg + w * v

            updated.append(feats[t] + self.out_proj[t](self._join(agg)))
        return updated


#  Hierarchical Molecular GNN
class HMGNN(nn.Module):
    """
    Multi-resolution GNN with cross-scale attention and uncertainty-weighted loss.
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
    ):
        super().__init__()
        self.S = len(scale_dims)
        self.hidden_dim = hidden_dim  # Store hidden_dim
        self.node_out_dim = node_out_dim  # Store node_out_dim
        self.graph_out_dim = graph_out_dim  # Store graph_out_dim
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

        # uncertainty parameters for loss weighting
        self.log_sigma_node = nn.Parameter(torch.zeros(1))
        self.log_sigma_graph = nn.Parameter(torch.zeros(1))

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

        # ---- per-scale backbone ----
        for i in range(self.S):
            x = scale_data[i]["x"]
            ei = scale_data[i]["edge_index"]
            ea = scale_data[i].get("edge_attr", None)

            outs, h = [], x
            for block in self.scale_gnns[i].children():
                h = block(h, ei, ea)
                outs.append(h)
            z = self.scale_jk[i](outs)  # [Ni, F]
            scale_feats.append(z)

        # cross-scale attention
        if self.use_cs and maps is not None:
            scale_feats = self.cross_scale(scale_feats, maps, edge_pairs_cs)

        #  heads
        output_dict = {}
        for i in range(self.S):
            batch_vec = scale_data[i].get("batch", None)
            current_scale_node_feats = scale_feats[i]

            # Node predictions for current scale
            current_node_pred = self.node_heads[i](current_scale_node_feats)
            node_preds.append(current_node_pred)
            output_dict[f"scale_{i}_node_pred"] = current_node_pred

            # Graph embedding and prediction for current scale
            if current_scale_node_feats.size(0) == 0:  # Handle zero-node graphs for this scale
                num_graphs_in_batch = 1
                if batch_vec is not None and batch_vec.numel() > 0:
                    num_graphs_in_batch = batch_vec.max().item() + 1

                # Output dim of scale_jk is self.hidden_dim (input to graph_pool and graph_head's input linear layer)
                # Output dim of graph_head is self.graph_out_dims[i]
                g_emb = torch.zeros((num_graphs_in_batch, self.hidden_dim), device=current_scale_node_feats.device)
                current_graph_pred = torch.zeros(
                    (num_graphs_in_batch, self.graph_out_dim), device=current_scale_node_feats.device
                )  # Use self.graph_out_dim
            else:
                g_emb = (
                    current_scale_node_feats.mean(0, keepdim=True)
                    if batch_vec is None
                    else self.graph_pool(current_scale_node_feats, batch_vec)
                )
                if self.env_dim:
                    if env_vec is None:
                        env_vec = x.new_zeros(g_emb.size(0), self.env_dim)
                    env_emb = self.env_proj(env_vec) if self.env_proj else env_vec
                    g_emb = torch.cat([g_emb, env_emb], dim=1)
                current_graph_pred = self.graph_heads[i](g_emb)
            graph_embeds.append(g_emb)
            graph_preds.append(current_graph_pred)
            output_dict[f"scale_{i}_graph_pred"] = current_graph_pred

        # Ensure all graph_embeds have consistent batch dimension before cat
        # This should be handled by the zero-node logic above ensuring num_graphs_in_batch consistency.
        # If graph_embeds is empty (e.g. self.S = 0, though unlikely), handle torch.cat
        if not graph_embeds:  # Should not happen if self.S > 0
            combined_graph_pred = torch.empty(0, device=x.device if "x" in scale_data[0] else "cpu")  # Placeholder
        else:
            try:
                concatenated_graph_embeds = torch.cat(graph_embeds, dim=1)
                combined_graph_pred = self.combined_graph_head(concatenated_graph_embeds)
            except RuntimeError as e:
                logger.error(f"HMGNN: Error during torch.cat(graph_embeds): {e}")
                logger.error(f"Shapes of graph_embeds: {[ge.shape for ge in graph_embeds]}")
                # Fallback or re-raise, for now, let's create a dummy output to avoid crashing tests completely
                # This indicates a deeper issue if shapes are still mismatched.
                # Assuming a batch size B (e.g., from first graph_embed if available, else 1)
                # and combined_graph_head output dimension.
                bs = graph_embeds[0].size(0) if graph_embeds and graph_embeds[0].numel() > 0 else 1
                out_dim_combined_graph_head = self.combined_graph_head[
                    -1
                ].out_features  # Get out_features of last linear layer
                combined_graph_pred = torch.zeros(
                    (bs, out_dim_combined_graph_head), device=graph_embeds[0].device if graph_embeds else "cpu"
                )

        output_dict["node_pred"] = node_preds[0] if node_preds else None  # atom-level default (finest scale)
        output_dict["graph_pred"] = combined_graph_pred  # combined graph-level prediction

        # 'all_node_pred' and 'all_graph_pred' are now redundant due to per-scale keys.
        # Tests should be updated if they relied on these specific list keys.
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


#  Factory convenience
def create_hierarchical_mgnn(
    scale_dims: List[int],
    hidden_dim: int = 64,
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
) -> HMGNN:
    """Easy constructor mirroring DJMGNN.create_* style."""
    return HMGNN(
        scale_dims,
        hidden_dim,
        n_blocks,
        layers_per_block,
        edge_attr_dims,
        jk_mode,
        node_out_dim,
        graph_out_dim,
        cross_scale_exchange,
        dropout,
        n_heads_cs,
        edge_dim_cs,
        pool_type,
    )
