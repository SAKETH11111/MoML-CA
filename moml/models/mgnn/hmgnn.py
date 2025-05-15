"""
hmgnn.py  – Hierarchical Molecular Graph Neural Network (HMGNN)

This file holds only the multi-scale / cross-scale machinery.  Core blocks such as
DenseGNNBlock and JKAggregator are reused from `djmgnn.py`.


"""

from __future__ import annotations
import math
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_scatter import scatter_add

from moml.models.mgnn.djmgnn import DenseGNNBlock, JKAggregator   # adjust the path if needed



#  GPU scatter / gather helpers
def aggregate_fine_to_coarse(feat: torch.Tensor,
                             fine2coarse: torch.Tensor,
                             coarse_count: torch.Tensor) -> torch.Tensor:
    """Mean-pool fine-scale features to coarse clusters."""
    summed = scatter_add(feat, fine2coarse, dim=0)
    return summed / coarse_count.unsqueeze(-1).clamp(min=1)


def broadcast_coarse_to_fine(coarse_feat: torch.Tensor,
                             fine2coarse: torch.Tensor) -> torch.Tensor:
    """Repeat coarse features on their member fine nodes."""
    return coarse_feat[fine2coarse]


#  Edge-conditioned one-step convolution on bipartite cross-scale graph
class EdgeNNConv(nn.Module):
    def __init__(self, in_src: int, in_tgt: int, out_dim: int, edge_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.nn = nn.Sequential(nn.Linear(edge_dim, in_src * out_dim),
                                nn.ReLU())

    def forward(self, x_src, x_tgt, edge_index, edge_attr):
        """
        edge_index : [2, E]  (row 0 = source indices, row 1 = target indices)
        edge_attr  : [E, edge_dim]
        """
        w = self.nn(edge_attr).view(-1, self.out_dim, x_src.size(1))     # [E, out, in_src]
        msg = torch.bmm(w, x_src[edge_index[0]].unsqueeze(-1)).squeeze(-1)  # [E, out]
        return scatter_add(msg, edge_index[1], dim=0, dim_size=x_tgt.size(0))


#  Multi-head cross-scale attention
class CrossScaleAttentionMH(nn.Module):
    """
    Multi-head attention across S scales (0 = finest).
    Optionally preceded by an EdgeNNConv message pass on each fine→coarse pair.
    """
    def __init__(self,
                 n_scales: int,
                 hidden_dim: int,
                 n_heads: int = 4,
                 edge_dim: int = 0):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.S, self.h, self.d_k = n_scales, n_heads, hidden_dim // n_heads
        self.scale = nn.Parameter(torch.tensor(math.sqrt(self.d_k)))     # learnable temperature

        make_proj = lambda: nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                          for _ in range(n_scales))
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = \
            make_proj(), make_proj(), make_proj(), make_proj()

        self.edge_msg = nn.ModuleList([
            EdgeNNConv(hidden_dim, hidden_dim, hidden_dim, edge_dim)
            for _ in range(n_scales - 1)
        ]) if edge_dim else None

    # helpers
    def _split(self, x):  return x.view(x.size(0), self.h, self.d_k)
    def _join (self, x):  return x.view(x.size(0), self.h * self.d_k)

    def forward(self,
                feats: List[torch.Tensor],                       # per-scale node embeddings
                maps: Tuple[List[torch.Tensor], List[torch.Tensor]],
                edge_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
               ) -> List[torch.Tensor]:
        fine2coarse, coarse_count = maps  # each length S-1 (scale i → i+1)

        # optional edge-conditioned messages (fine → coarse)
        if self.edge_msg and edge_pairs is not None:
            for i in range(self.S - 1):
                ei, ea = edge_pairs[i]
                msg = self.edge_msg[i](feats[i], feats[i+1], ei, ea)
                feats[i+1] = feats[i+1] + msg

        updated = []
        for t in range(self.S):                                     # target scale
            q = self._split(self.q_proj[t](feats[t]))               # [Nt, h, d_k]
            agg = torch.zeros_like(q)

            for s in range(self.S):                                 # source scale
                k = self._split(self.k_proj[s](feats[s]))
                v = self._split(self.v_proj[s](feats[s]))

                if s != t:
                    if s < t:    # fine → coarse: aggregate
                        k = aggregate_fine_to_coarse(k, fine2coarse[s], coarse_count[s])
                        v = aggregate_fine_to_coarse(v, fine2coarse[s], coarse_count[s])
                    else:        # coarse → fine: broadcast
                        k = broadcast_coarse_to_fine(k, fine2coarse[t])
                        v = broadcast_coarse_to_fine(v, fine2coarse[t])

                scores = (q * k).sum(-1) / self.scale               # [Nt, h]
                w = scores.softmax(-1).unsqueeze(-1)                # [Nt, h, 1]
                agg = agg + w * v

            updated.append(feats[t] + self.out_proj[t](self._join(agg)))
        return updated


#  Hierarchical Molecular GNN
class HMGNN(nn.Module):
    """
    Multi-resolution GNN with cross-scale attention and uncertainty-weighted loss.
    """
    def __init__(self,
                 scale_dims: List[int],
                 hidden_dim: int = 64,
                 n_blocks: int = 2,
                 layers_per_block: int = 3,
                 edge_attr_dims: Optional[List[int]] = None,
                 jk_mode: str = 'attention',
                 node_out_dim: int = 1,
                 graph_out_dim: int = 1,
                 cross_scale_exchange: bool = True,
                 dropout: float = 0.2,
                 n_heads_cs: int = 4,
                 edge_dim_cs: int = 0,
                 pool_type: str = 'mean'):
        super().__init__()
        self.S = len(scale_dims)
        edge_attr_dims = edge_attr_dims or [0] * self.S

        # backbone per scale
        self.scale_gnns, self.scale_jk = nn.ModuleList(), nn.ModuleList()
        for d_in, e_dim in zip(scale_dims, edge_attr_dims):
            blocks, dims = nn.ModuleList(), []
            # first block
            blocks.append(DenseGNNBlock(d_in, hidden_dim, layers_per_block,
                                        hidden_dim, e_dim))
            dims.append(hidden_dim)
            # subsequent blocks
            for _ in range(n_blocks - 1):
                blocks.append(DenseGNNBlock(hidden_dim, hidden_dim,
                                            layers_per_block, hidden_dim, e_dim))
                dims.append(hidden_dim)
            self.scale_gnns.append(blocks)
            self.scale_jk.append(JKAggregator(dims, hidden_dim, mode=jk_mode))

        # cross-scale module
        self.use_cs = cross_scale_exchange
        if cross_scale_exchange:
            self.cross_scale = CrossScaleAttentionMH(self.S, hidden_dim,
                                                     n_heads_cs, edge_dim_cs)

        # heads
        self.node_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
                          nn.Dropout(dropout),
                          nn.Linear(hidden_dim//2, node_out_dim))
            for _ in range(self.S)
        ])
        pool_dict = {'mean': global_mean_pool, 'add': global_add_pool, 'max': global_max_pool}
        self.graph_pool = pool_dict.get(pool_type, global_mean_pool)
        self.graph_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
                          nn.Dropout(dropout),
                          nn.Linear(hidden_dim//2, graph_out_dim))
            for _ in range(self.S)
        ])
        self.combined_graph_head = nn.Sequential(
            nn.Linear(hidden_dim * self.S, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, graph_out_dim)
        )

        # uncertainty parameters for loss weighting
        self.log_sigma_node = nn.Parameter(torch.zeros(1))
        self.log_sigma_graph = nn.Parameter(torch.zeros(1))

    def forward(self,
                scale_data: List[Dict[str, Any]],
                maps: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
                edge_pairs_cs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
               ) -> Dict[str, Any]:
        """
        scale_data[i] = {'x', 'edge_index', 'edge_attr'?, 'batch'?, 'mask'?}
        maps          = (fine2coarse, coarse_count)   – each list length S-1
        edge_pairs_cs = list length S-1 of (edge_index, edge_attr) for fine→coarse
        """
        scale_feats, graph_embeds, node_preds, graph_preds = [], [], [], []

        # ---- per-scale backbone ----
        for i in range(self.S):
            x = scale_data[i]['x']
            ei = scale_data[i]['edge_index']
            ea = scale_data[i].get('edge_attr', None)

            outs, h = [], x
            for block in self.scale_gnns[i].children():
                h = block(h, ei, ea)
                outs.append(h)
            z = self.scale_jk[i](outs)           # [Ni, F]
            scale_feats.append(z)

        # cross-scale attention 
        if self.use_cs and maps is not None:
            scale_feats = self.cross_scale(scale_feats, maps, edge_pairs_cs)

        #  heads
        for i in range(self.S):
            batch = scale_data[i].get('batch', None)
            node_preds.append(self.node_heads[i](scale_feats[i]))

            g_emb = (scale_feats[i].mean(0, keepdim=True) if batch is None
                     else self.graph_pool(scale_feats[i], batch))
            graph_embeds.append(g_emb)
            graph_preds.append(self.graph_heads[i](g_emb))

        combined_graph_pred = self.combined_graph_head(torch.cat(graph_embeds, 1))

        return {'node_pred': node_preds[0],             # atom-level default
                'graph_pred': combined_graph_pred,
                'all_node_pred': node_preds,
                'all_graph_pred': graph_preds}

    def compute_losses(self,
                       preds: Dict[str, Any],
                       targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        targets = {'node': (y, mask) , 'graph': (y, mask)}
        """
        y_n, m_n = targets['node']
        y_g, m_g = targets['graph']

        node_loss  = F.mse_loss(preds['node_pred'][m_n],  y_n, reduction='mean')
        graph_loss = F.mse_loss(preds['graph_pred'][m_g], y_g, reduction='mean')

        total = (node_loss  / torch.exp(self.log_sigma_node)  + self.log_sigma_node) + \
                (graph_loss / torch.exp(self.log_sigma_graph) + self.log_sigma_graph)
        return total, {'node': node_loss.item(), 'graph': graph_loss.item()}



#  Factory convenience
def create_hierarchical_mgnn(scale_dims: List[int],
                             hidden_dim: int = 64,
                             n_blocks: int = 2,
                             layers_per_block: int = 3,
                             edge_attr_dims: Optional[List[int]] = None,
                             jk_mode: str = 'attention',
                             node_out_dim: int = 1,
                             graph_out_dim: int = 1,
                             cross_scale_exchange: bool = True,
                             dropout: float = 0.2,
                             n_heads_cs: int = 4,
                             edge_dim_cs: int = 0,
                             pool_type: str = 'mean'
                            ) -> HMGNN:
    """Easy constructor mirroring DJMGNN.create_* style."""
    return HMGNN(scale_dims, hidden_dim, n_blocks, layers_per_block,
                 edge_attr_dims, jk_mode,
                 node_out_dim, graph_out_dim,
                 cross_scale_exchange, dropout,
                 n_heads_cs, edge_dim_cs, pool_type)
