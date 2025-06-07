import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, global_max_pool, GraphNorm
import logging

logger = logging.getLogger(__name__)


# helpers
def rbf_encode_dist(dists, K=32, d_min=0.0, d_max=10.0):
    """Gaussian RBF encoding of distances (shape: [num_edges, 1] → [num_edges, K])."""
    mu = torch.linspace(d_min, d_max, K, device=dists.device)
    gamma = -0.5 / ((mu[1] - mu[0]) ** 2)
    diff = dists - mu.view(1, -1)
    return torch.exp(gamma * diff**2)


# core layers
class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super().__init__()
        self.actual_edge_attr_dim = edge_attr_dim
        _mlp_input_dim = 1 if edge_attr_dim == 0 else edge_attr_dim

        self.edge_mlp = nn.Sequential(nn.Linear(_mlp_input_dim, in_channels * out_channels), nn.ReLU())
        self.conv = NNConv(in_channels, out_channels, nn=self.edge_mlp, aggr="add")
        self.norm = GraphNorm(out_channels)
        self.res_connection = in_channels == out_channels

    def forward(self, x, edge_index, edge_attr):
        edge_attr_for_nnconv_input = edge_attr

        if self.actual_edge_attr_dim == 0:
            edge_attr_for_nnconv_input = None
        elif edge_attr is None and self.actual_edge_attr_dim > 0:
            edge_attr_for_nnconv_input = None

        if edge_attr_for_nnconv_input is None and edge_index.numel() > 0:
            dummy_dim = self.edge_mlp[0].in_features if hasattr(self.edge_mlp[0], 'in_features') else self.actual_edge_attr_dim
            edge_attr_for_nnconv_input = x.new_ones(edge_index.size(1), dummy_dim)
        elif edge_index.numel() == 0:
            edge_attr_for_nnconv_input = torch.empty(
                0, self.edge_mlp[0].in_features if hasattr(self.edge_mlp[0], "in_features") else 1
            ).to(x.device)

        h = self.conv(x, edge_index, edge_attr_for_nnconv_input)
        h = self.norm(h)
        h = F.relu(h)

        if self.res_connection:
            h = h + x
        return h


class DenseGNNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, transition_dim, edge_attr_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.initial_proj = nn.Linear(in_dim, hidden_dim)
        
        self.conv_layers = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.conv_layers.append(GraphConvLayer(hidden_dim, hidden_dim, edge_attr_dim))
            # Transition layer to process concatenated features
            self.transition_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))

        self.final_transition = nn.Linear(hidden_dim, transition_dim)
        self.norm = GraphNorm(transition_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.initial_proj(x)
        
        for i in range(self.n_layers):
            h_conv = self.conv_layers[i](h, edge_index, edge_attr)
            h_cat = torch.cat([h, h_conv], 1)
            h = self.transition_layers[i](h_cat)
            h = F.relu(h)
            
        h_final = self.final_transition(h)
        return F.relu(self.norm(h_final))


class JKAggregator(nn.Module):
    def __init__(self, block_dims, out_dim, mode="attention"):
        super().__init__()
        self.mode, self.block_count = mode, len(block_dims)

        if mode == "concat":
            if not block_dims:
                raise ValueError("block_dims cannot be empty for concat mode.")
            self.proj = nn.Linear(sum(block_dims), out_dim)
        elif mode == "max":
            if not block_dims:
                raise ValueError("block_dims cannot be empty for max mode.")
            self.projs = nn.ModuleList(nn.Linear(d, out_dim) for d in block_dims)
        elif mode == "attention":
            if not block_dims:
                raise ValueError("block_dims cannot be empty for attention mode.")
            self.projs = nn.ModuleList(nn.Linear(d, out_dim) for d in block_dims)
            self.attn_vecs = nn.ParameterList(nn.Parameter(torch.randn(out_dim)) for _ in range(self.block_count))
        elif mode == "lstm":
            if not block_dims:
                self.lstm_input_dim = out_dim
                self.lstm_projs_in = nn.ModuleList()
                self.lstm_layer = nn.LSTM(
                    input_size=self.lstm_input_dim, hidden_size=self.lstm_input_dim, num_layers=1, batch_first=False
                )
                self.lstm_final_proj = nn.Linear(self.lstm_input_dim, out_dim)
            else:
                self.lstm_input_dim = out_dim
                self.lstm_projs_in = nn.ModuleList(nn.Linear(d, self.lstm_input_dim) for d in block_dims)
                self.lstm_layer = nn.LSTM(
                    input_size=self.lstm_input_dim, hidden_size=self.lstm_input_dim, num_layers=1, batch_first=False
                )
                self.lstm_final_proj = nn.Linear(self.lstm_input_dim, out_dim)
        else:
            raise ValueError(f"Unsupported JKAggregator mode: {mode}")

        _fallback_in_dim = sum(block_dims) if block_dims else out_dim
        if _fallback_in_dim == 0:
            _fallback_in_dim = out_dim if out_dim > 0 else 1
        self.fallback_proj = nn.Linear(_fallback_in_dim, out_dim)

    def forward(self, blocks):
        if not blocks:
            if self.mode == "lstm" and self.block_count == 0:
                return None
            return None

        if self.mode == "concat":
            return self.proj(torch.cat(blocks, 1))
        elif self.mode == "max":
            projected_blocks = [proj(block) for proj, block in zip(self.projs, blocks)]
            if not projected_blocks:
                return self.fallback_proj(
                    torch.cat(blocks, 1)
                    if blocks
                    else torch.empty(0, self.fallback_proj.in_features).to(self.fallback_proj.weight.device)
                )
            return torch.max(torch.stack(projected_blocks, 0), 0)[0]
        elif self.mode == "attention":
            projected_blocks = [proj(block) for proj, block in zip(self.projs, blocks)]
            if not projected_blocks:
                return self.fallback_proj(
                    torch.cat(blocks, 1)
                    if blocks
                    else torch.empty(0, self.fallback_proj.in_features).to(self.fallback_proj.weight.device)
                )
            scores = [(b * v).sum(-1) / math.sqrt(b.size(-1)) for b, v in zip(projected_blocks, self.attn_vecs)]
            w = torch.stack(scores, 1).softmax(1)
            return sum(w[:, i : i + 1] * projected_blocks[i] for i in range(self.block_count))
        elif self.mode == "lstm":
            if not self.block_count > 0:
                return self.fallback_proj(
                    torch.zeros(
                        blocks[0].size(0) if blocks and blocks[0].numel() > 0 else 0, self.fallback_proj.in_features
                    ).to(self.fallback_proj.weight.device)
                )

            try:
                projected_blocks = [proj(block) for proj, block in zip(self.lstm_projs_in, blocks)]
                num_nodes_first_block = projected_blocks[0].size(0)
                for i, p_block in enumerate(projected_blocks):
                    if p_block.size(0) != num_nodes_first_block:
                        raise ValueError(
                            f"LSTM mode: All blocks must have the same number of nodes. Block 0: {num_nodes_first_block}, Block {i}: {p_block.size(0)}"
                        )
                    if p_block.size(1) != self.lstm_input_dim:
                        raise ValueError(
                            f"LSTM mode: Projected block {i} has incorrect feature dimension. Expected {self.lstm_input_dim}, got {p_block.size(1)}"
                        )

                stacked_blocks = torch.stack(projected_blocks, dim=0)
                lstm_out, _ = self.lstm_layer(stacked_blocks)
                last_layer_sequence_output = lstm_out[-1, :, :]
                return self.lstm_final_proj(last_layer_sequence_output)
            except Exception as e:
                logger.error(f"Error in JKAggregator LSTM forward: {e}. Using fallback.")
                return self.fallback_proj(torch.cat(blocks, 1))

        logger.warning(f"JKAggregator: Unhandled mode '{self.mode}' in forward. Using fallback_proj.")
        return self.fallback_proj(torch.cat(blocks, 1))


class DJMGNN(nn.Module):
    def __init__(
        self,
        in_node_dim, 
        hidden_dim,
        n_blocks=3,
        layers_per_block=6,
        in_edge_dim=0,  
        jk_mode="attention",
        node_output_dims=3, 
        graph_output_dims=19, 
        energy_output_dims=1,
        dropout=0.2, 
        pool_type="mean",
        p_dropedge=0.1,
        use_supernode=True,
        use_rbf=True,
        rbf_K=32,
    ):
        super().__init__()
        self.p_dropedge, self.use_super, self.use_rbf, self.rbf_K = p_dropedge, use_supernode, use_rbf, rbf_K
        self.hidden_dim = hidden_dim
        self.node_output_dims = node_output_dims  
        self.graph_output_dims = graph_output_dims
        self.energy_output_dims = energy_output_dims

        self.input_edge_attr_dim = in_edge_dim  
        self.processed_edge_attr_dim = self.input_edge_attr_dim + (self.rbf_K if self.use_rbf else 0)

        self.blocks = nn.ModuleList()
        current_block_in_dim = in_node_dim
        for _ in range(n_blocks):
            self.blocks.append(
                DenseGNNBlock(
                    in_dim=current_block_in_dim,
                    hidden_dim=hidden_dim,
                    n_layers=layers_per_block,
                    transition_dim=hidden_dim,
                    edge_attr_dim=self.processed_edge_attr_dim,
                )
            )
            current_block_in_dim = hidden_dim

        self.jk = JKAggregator([hidden_dim] * n_blocks, hidden_dim, mode=jk_mode)

        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_output_dims)
        )
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, graph_output_dims)
        )
        self.head_energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, energy_output_dims)
        )
        self.pool = {"mean": global_mean_pool, "add": global_add_pool, "max": global_max_pool}.get(
            pool_type, global_mean_pool
        )

    def add_supernode(self, x, edge_index, edge_attr, batch):
        if not self.use_super or x.numel() == 0:
            return x, edge_index, edge_attr, batch

        num_nodes_original = x.size(0)
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 0
        if num_graphs == 0 and num_nodes_original > 0:
            num_graphs = 1
        elif num_graphs == 0 and num_nodes_original == 0:
            return x, edge_index, edge_attr, batch

        super_feat = x.new_zeros((num_graphs, x.size(1)))
        x_with_super = torch.cat([x, super_feat], 0)

        device = x.device
        row = torch.arange(num_nodes_original, device=device)
        col_batch_indices = (
            batch[:num_nodes_original]
            if batch.numel() >= num_nodes_original
            else torch.zeros(num_nodes_original, dtype=torch.long, device=device)
        )

        col_supernode_indices = col_batch_indices + num_nodes_original

        edge1 = torch.stack([row, col_supernode_indices], 0)
        edge2 = torch.stack([col_supernode_indices, row], 0)

        new_edge_index = torch.cat([edge_index, edge1, edge2], 1)
        new_edge_attr = edge_attr

        if edge_attr is not None:
            if edge_attr.numel() > 0:
                super_e = edge_attr.new_zeros(edge1.size(1) + edge2.size(1), edge_attr.size(1))
                new_edge_attr = torch.cat([edge_attr, super_e], 0)
            elif self.processed_edge_attr_dim > 0:
                super_e = x.new_zeros(edge1.size(1) + edge2.size(1), self.processed_edge_attr_dim)
                if edge_attr is None:
                    new_edge_attr = super_e
                if new_edge_attr is None and self.processed_edge_attr_dim > 0:
                    pass

        batch_for_original_nodes = batch[:num_nodes_original]
        batch_for_super_nodes = torch.arange(num_graphs, device=device)
        final_new_batch = torch.cat([batch_for_original_nodes, batch_for_super_nodes], dim=0)

        return x_with_super, new_edge_index, new_edge_attr, final_new_batch

    def drop_edges(self, edge_index, edge_attr):
        if not self.training or self.p_dropedge == 0:
            return edge_index, edge_attr
        if edge_index.numel() == 0:
            return edge_index, edge_attr
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.p_dropedge
        return edge_index[:, mask], (edge_attr[mask] if edge_attr is not None and edge_attr.numel() > 0 else edge_attr)

    def forward(self, x, edge_index, edge_attr=None, batch=None, dist=None):
        if x is None or x.numel() == 0:
            return {
                "node_pred": torch.empty(0, self.node_output_dims).to(x.device), 
                "graph_pred": torch.empty(0, self.graph_output_dims).to(x.device), 
                "energy_pred": torch.empty(0, self.energy_output_dims).to(x.device)
            }

        num_edges_initial = edge_index.size(1)
        
        original_feat_part = None
        if self.input_edge_attr_dim > 0:
            if edge_attr is not None:
                if edge_attr.size(0) == num_edges_initial and edge_attr.size(1) == self.input_edge_attr_dim:
                    original_feat_part = edge_attr
                else:
                    logger.warning(
                        f"DJMGNN: input edge_attr shape {edge_attr.shape} mismatch with expected ({num_edges_initial}, {self.input_edge_attr_dim}). Using zeros for original part."
                    )
                    original_feat_part = torch.zeros(num_edges_initial, self.input_edge_attr_dim, device=x.device)
            else:
                original_feat_part = torch.zeros(num_edges_initial, self.input_edge_attr_dim, device=x.device)
        
        rbf_feat_part = None
        if self.use_rbf:
            rbf_k_feats = torch.zeros(num_edges_initial, self.rbf_K, device=x.device)
            if dist is not None and dist.numel() > 0:
                if dist.size(0) == num_edges_initial:
                    rbf_k_feats = rbf_encode_dist(dist, K=self.rbf_K)
                else:
                    logger.warning(
                        f"DJMGNN: dist size {dist.size(0)} mismatch with edge_index size {num_edges_initial}. Using zero RBF features."
                    )
            rbf_feat_part = rbf_k_feats

        if original_feat_part is not None and rbf_feat_part is not None:
            current_edge_attr = torch.cat([original_feat_part, rbf_feat_part], dim=1)
        elif original_feat_part is not None:
            current_edge_attr = original_feat_part
        elif rbf_feat_part is not None:
            current_edge_attr = rbf_feat_part
        else:
            current_edge_attr = None

        if current_edge_attr is not None:
            if current_edge_attr.size(1) != self.processed_edge_attr_dim:
                logger.error(
                    f"DJMGNN: FATAL current_edge_attr dim {current_edge_attr.size(1)} mismatch with processed_edge_attr_dim {self.processed_edge_attr_dim}. This indicates a bug."
                )
        elif self.processed_edge_attr_dim > 0:
             logger.warning(
                f"DJMGNN: current_edge_attr is None, but processed_edge_attr_dim is {self.processed_edge_attr_dim}. Creating zeros."
             )
             if num_edges_initial > 0:
                 current_edge_attr = torch.zeros(num_edges_initial, self.processed_edge_attr_dim, device=x.device)

        current_batch = batch if batch is not None else x.new_zeros(x.size(0), dtype=torch.long)

        current_x, current_edge_index, current_edge_attr, current_batch = self.add_supernode(
            x, edge_index, current_edge_attr, current_batch
        )

        current_edge_index, current_edge_attr = self.drop_edges(current_edge_index, current_edge_attr)

        h_intermediate, outs = current_x, []
        for block in self.blocks:
            if h_intermediate.numel() == 0:
                block_output_dim = block.final_transition.out_features if hasattr(block, "final_transition") else self.hidden_dim
                h_intermediate = torch.empty(0, block_output_dim).to(x.device)
            else:
                h_intermediate = block(h_intermediate, current_edge_index, current_edge_attr)
            outs.append(h_intermediate)

        h_aggregated = self.jk(outs)

        if h_aggregated is None or h_aggregated.numel() == 0:
            num_output_nodes = 0
            batch_size_for_graph_pred = current_batch.max().item() + 1 if current_batch.numel() > 0 else 0
            out_node = torch.empty(num_output_nodes, self.node_output_dims).to(x.device) 
            out_graph = torch.empty(batch_size_for_graph_pred, self.graph_output_dims).to(x.device) 
            out_energy = torch.empty(batch_size_for_graph_pred, self.energy_output_dims).to(x.device)
            return {"node_pred": out_node, "graph_pred": out_graph, "energy_pred": out_energy}

        num_original_nodes = x.size(0) 
        
        if h_aggregated.size(0) > num_original_nodes and self.use_super:
            node_emb_for_head = h_aggregated[:num_original_nodes]
        else:
            node_emb_for_head = h_aggregated
            if h_aggregated.size(0) < num_original_nodes:
                 logger.warning(f"h_aggregated size {h_aggregated.size(0)} is smaller than num_original_nodes {num_original_nodes}. This might be unexpected.")

        out_node = self.node_head(node_emb_for_head)

        graph_emb_input = h_aggregated 
        graph_emb = self.pool(graph_emb_input, current_batch)
        
        out_graph = self.graph_head(graph_emb)
        out_energy = self.head_energy(graph_emb)
        
        return {"node_pred": out_node, "graph_pred": out_graph, "energy_pred": out_energy}
