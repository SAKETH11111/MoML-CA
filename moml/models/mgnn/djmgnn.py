import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, global_max_pool, GraphNorm
import logging  # Added for logging potential issues

logger = logging.getLogger(__name__)  # Added


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

        # If edge_attr_for_nnconv_input is None at this point, NNConv will create a dummy [E,1] tensor.
        # Our self.edge_mlp (Linear(1,K) if actual_edge_attr_dim == 0) is set up for this.
        # If actual_edge_attr_dim > 0 and edge_attr was None, NNConv's dummy [E,1] might mismatch
        # self.edge_mlp if it expected >1 features. This indicates an upstream issue.
        # The test `test_forward_pass_no_edge_attr` passes None when actual_edge_attr_dim is 0.
        if edge_attr_for_nnconv_input is None and edge_index.numel() > 0:  # Ensure we create dummy only if edges exist
            # Create dummy edge attributes matching expected dimension
            dummy_dim = self.edge_mlp[0].in_features if hasattr(self.edge_mlp[0], 'in_features') else self.actual_edge_attr_dim
            edge_attr_for_nnconv_input = x.new_ones(edge_index.size(1), dummy_dim)
        elif edge_index.numel() == 0:  # No edges, edge_attr should be empty or None
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
        self.in_dim = in_dim  # Store in_dim
        self.layers, cur_dim = nn.ModuleList(), in_dim
        for _ in range(n_layers):
            self.layers.append(GraphConvLayer(cur_dim, hidden_dim, edge_attr_dim))
            cur_dim += hidden_dim
        self.transition = nn.Linear(cur_dim, transition_dim)
        self.norm = GraphNorm(transition_dim)

    def forward(self, x, edge_index, edge_attr):
        outs = [x]
        for layer in self.layers:
            h = layer(torch.cat(outs, 1), edge_index, edge_attr)
            outs.append(h)
        h_concat = torch.cat(outs, 1)
        h_transition = self.transition(h_concat)
        return F.relu(self.norm(h_transition))


class JKAggregator(nn.Module):
    def __init__(self, block_dims, out_dim, mode="attention"):
        super().__init__()
        self.mode, self.block_count = mode, len(block_dims)

        # Removed generic check, mode-specific checks will handle empty block_dims

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
            if not block_dims:  # If 0 blocks, LSTM acts as a simple projection from a zero vector or configured input
                # This case is ill-defined for standard JK-LSTM.
                # We'll define components so it doesn't crash, but it won't be a meaningful LSTM aggregation.
                self.lstm_input_dim = out_dim  # Dummy
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
            _fallback_in_dim = out_dim if out_dim > 0 else 1  # Ensure non-zero for Linear
        self.fallback_proj = nn.Linear(_fallback_in_dim, out_dim)

    def forward(self, blocks):
        if not blocks:
            if self.mode == "lstm" and self.block_count == 0:  # LSTM initialized for 0 blocks
                # Requires careful thought on what to return. For now, None or zeros.
                # Assuming out_dim is known. Need num_nodes for batch.
                # This path is highly dependent on how DJMGNN handles zero-node/zero-block graphs.
                # Returning None will likely cause downstream errors, which is fine for now to highlight the issue.
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
                )  # Should not happen if blocks not empty
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
            if not self.block_count > 0:  # Handle case where LSTM was init with 0 blocks
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
        in_dim,
        hidden_dim,
        n_blocks=3,
        layers_per_block=6,
        edge_attr_dim=0,  # This is the dimension of the *input* edge_attr, without RBF
        jk_mode="attention",
        node_out_dim=1,
        graph_out_dim=1,
        dropout=0.2,
        pool_type="mean",
        p_dropedge=0.1,
        use_supernode=True,
        use_rbf=True,
        rbf_K=32,
    ):
        super().__init__()
        self.p_dropedge, self.use_super, self.use_rbf, self.rbf_K = p_dropedge, use_supernode, use_rbf, rbf_K

        self.input_edge_attr_dim = edge_attr_dim  # Store original input edge_attr_dim

        # Dimension of edge attributes after potentially adding RBF features
        self.processed_edge_attr_dim = self.input_edge_attr_dim + (self.rbf_K if self.use_rbf else 0)

        self.blocks = nn.ModuleList()
        current_block_in_dim = in_dim
        for _ in range(n_blocks):
            self.blocks.append(
                DenseGNNBlock(
                    in_dim=current_block_in_dim,
                    hidden_dim=hidden_dim,
                    n_layers=layers_per_block,
                    transition_dim=hidden_dim,  # Output of DenseGNNBlock's transition layer
                    edge_attr_dim=self.processed_edge_attr_dim,  # Pass the final dim to blocks
                )
            )
            # Input to next block is output of current block's transition layer
            current_block_in_dim = hidden_dim  # As transition_dim is hidden_dim

        self.jk = JKAggregator([hidden_dim] * n_blocks, hidden_dim, mode=jk_mode)

        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, node_out_dim),
        )
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, graph_out_dim),
        )
        self.pool = {"mean": global_mean_pool, "add": global_add_pool, "max": global_max_pool}.get(
            pool_type, global_mean_pool
        )

    def add_supernode(self, x, edge_index, edge_attr, batch):
        if not self.use_super or x.numel() == 0:  # also check x.numel()
            return x, edge_index, edge_attr, batch

        num_nodes_original = x.size(0)
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 0
        if num_graphs == 0 and num_nodes_original > 0:  # Single graph, no batch vector
            num_graphs = 1  # Assume one graph
        elif num_graphs == 0 and num_nodes_original == 0:  # Empty input
            return x, edge_index, edge_attr, batch

        super_feat = x.new_zeros((num_graphs, x.size(1)))
        x_with_super = torch.cat([x, super_feat], 0)

        device = x.device
        row = torch.arange(num_nodes_original, device=device)
        # Ensure batch corresponds to original nodes before supernode addition
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
            # Supernode edges have zero attributes of the same dimension as other edges
            if edge_attr.numel() > 0:
                super_e = edge_attr.new_zeros(edge1.size(1) + edge2.size(1), edge_attr.size(1))
                new_edge_attr = torch.cat([edge_attr, super_e], 0)
            # If original edge_attr was empty but had feature dim (e.g. from RBF only), create zeros
            elif self.processed_edge_attr_dim > 0:
                super_e = x.new_zeros(edge1.size(1) + edge2.size(1), self.processed_edge_attr_dim)
                # If original edge_attr was None but should have had features (e.g. RBF only)
                # This assumes edge_attr should have been zeros(0, dim) not None
                # For safety, if edge_attr is None, new_edge_attr remains None unless super_e is created
                if edge_attr is None:
                    new_edge_attr = super_e  # This might be wrong if original edges existed
                # This part is tricky if edge_attr is None but processed_edge_attr_dim > 0
                # Let's assume if edge_attr is None, new_edge_attr remains None and GraphConvLayer handles it.
                # The above cat would fail if edge_attr is None.
                # So, if edge_attr is None, new_edge_attr should also be None (GraphConvLayer will make dummy)
                # Or, if we want supernode edges to have *some* attr:
                if new_edge_attr is None and self.processed_edge_attr_dim > 0:
                    # Create dummy for all edges if original was None
                    all_zeros_for_all_edges = x.new_zeros(new_edge_index.size(1), self.processed_edge_attr_dim)
                    # This is not quite right, as GraphConvLayer makes a 1-dim dummy.
                    # Let's stick to: if original edge_attr is None, pass None. Supernode edges won't get explicit attrs.
                    pass

        # Correct batch assignment:
        # Original nodes: use their original batch indices
        # Supernodes: each supernode 'i' belongs to graph 'i'.
        batch_for_original_nodes = batch[:num_nodes_original]
        batch_for_super_nodes = torch.arange(num_graphs, device=device)  # Indices 0 to num_graphs-1
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
        if x.numel() == 0:
            # logger.warning("DJMGNN forward called with zero nodes.")
            return {
                "node_pred": torch.empty(0, self.node_head[-1].out_features).to(x.device),
                "graph_pred": torch.empty(0, self.graph_head[-1].out_features).to(x.device),
            }

        num_edges_initial = edge_index.size(1)
        current_edge_attr = edge_attr  # This is the input edge_attr (e.g. from SMILES, could be None or have features)

        if self.use_rbf:
            rbf_k_feats = torch.zeros(num_edges_initial, self.rbf_K, device=x.device)
            if dist is not None and dist.numel() > 0:
                if dist.size(0) == num_edges_initial:
                    rbf_k_feats = rbf_encode_dist(dist, K=self.rbf_K)
                else:
                    logger.warning(
                        f"DJMGNN: dist size {dist.size(0)} mismatch with edge_index size {num_edges_initial}. Using zero RBF features."
                    )

            if current_edge_attr is None:
                current_edge_attr = rbf_k_feats
            else:
                if current_edge_attr.size(0) == num_edges_initial:
                    current_edge_attr = torch.cat([current_edge_attr, rbf_k_feats], dim=1)
                else:  # Mismatch between current_edge_attr rows and num_edges_initial
                    logger.warning(
                        f"DJMGNN: input edge_attr rows {current_edge_attr.size(0)} mismatch with edge_index {num_edges_initial}. Reconstructing edge_attr with RBF."
                    )
                    # Determine expected original feature part
                    original_feat_dim = self.input_edge_attr_dim
                    original_part = torch.zeros(num_edges_initial, original_feat_dim, device=x.device)
                    # This assumes original_edge_attr was meant to be zeros if not provided correctly.
                    current_edge_attr = torch.cat([original_part, rbf_k_feats], dim=1)
        # At this point, current_edge_attr has features of dim:
        # self.input_edge_attr_dim + self.rbf_K (if use_rbf)
        # OR self.input_edge_attr_dim (if not use_rbf)
        # This matches self.processed_edge_attr_dim used to init blocks.

        current_batch = batch if batch is not None else x.new_zeros(x.size(0), dtype=torch.long)

        current_x, current_edge_index, current_edge_attr, current_batch = self.add_supernode(
            x, edge_index, current_edge_attr, current_batch
        )

        current_edge_index, current_edge_attr = self.drop_edges(current_edge_index, current_edge_attr)

        h_intermediate, outs = current_x, []
        for block_idx, block in enumerate(self.blocks):
            if h_intermediate.numel() == 0:
                block_output_dim = block.transition.out_features if hasattr(block, "transition") else self.hidden_dim
                h_intermediate = torch.empty(0, block_output_dim).to(x.device)
            else:
                h_intermediate = block(h_intermediate, current_edge_index, current_edge_attr)
            outs.append(h_intermediate)

        h_aggregated = self.jk(outs)

        if h_aggregated is None or h_aggregated.numel() == 0:
            num_output_nodes = 0
            batch_size_for_graph_pred = current_batch.max().item() + 1 if current_batch.numel() > 0 else 0
            node_pred = torch.empty(num_output_nodes, self.node_head[-1].out_features).to(x.device)
            graph_pred = torch.empty(batch_size_for_graph_pred, self.graph_head[-1].out_features).to(x.device)
            return {"node_pred": node_pred, "graph_pred": graph_pred}

        node_pred = self.node_head(h_aggregated)

        # Pooling requires valid batch vector that corresponds to h_aggregated
        # h_aggregated includes supernodes. current_batch also includes supernodes.
        if h_aggregated.size(0) == 0:
            graph_pooled = torch.zeros(
                current_batch.max().item() + 1 if current_batch.numel() > 0 else 0,
                h_aggregated.size(1) if h_aggregated.dim() > 1 else self.graph_head[0].in_features,
            ).to(h_aggregated.device if hasattr(h_aggregated, "device") else x.device)
        else:
            graph_pooled = self.pool(h_aggregated, current_batch)

        graph_pred = self.graph_head(graph_pooled)
        return {"node_pred": node_pred, "graph_pred": graph_pred}
