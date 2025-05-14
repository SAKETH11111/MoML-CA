import math, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, global_max_pool, GraphNorm

# helpers 
def rbf_encode_dist(dists, K=32, d_min=0.0, d_max=10.0):
    """Gaussian RBF encoding of distances (shape: [num_edges, 1] → [num_edges, K])."""
    mu = torch.linspace(d_min, d_max, K, device=dists.device)          # centres
    gamma = -0.5 / ((mu[1] - mu[0]) ** 2)                              # width
    diff = dists - mu.view(1, -1)                                      # broadcast
    return torch.exp(gamma * diff ** 2)                                # [E, K]

# core layers 
class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, in_channels * out_channels),
            nn.ReLU()
        )
        self.conv = NNConv(in_channels, out_channels, nn=self.edge_mlp, aggr='add')
        self.norm = GraphNorm(out_channels)             # BN → GraphNorm / LayerNorm
        self.res_connection = (in_channels == out_channels)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:                           # (C)
            edge_attr = x.new_ones(edge_index.size(1), 1)

        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h)
        h = F.relu(h)

        if self.res_connection:                        # (B)
            h = h + x
        return h

class DenseGNNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, transition_dim, edge_attr_dim):
        super().__init__()
        self.layers, cur_dim = nn.ModuleList(), in_dim
        for _ in range(n_layers):
            self.layers.append(GraphConvLayer(cur_dim, hidden_dim, edge_attr_dim))
            cur_dim += hidden_dim                       # dense concat
        self.transition = nn.Linear(cur_dim, transition_dim)
        self.norm = GraphNorm(transition_dim)

    def forward(self, x, edge_index, edge_attr):
        outs = [x]
        for layer in self.layers:
            h = layer(torch.cat(outs, 1), edge_index, edge_attr)
            outs.append(h)
        h = self.transition(torch.cat(outs, 1))
        return F.relu(self.norm(h))

class JKAggregator(nn.Module):
    def __init__(self, block_dims, out_dim, mode='attention'):
        super().__init__()
        self.mode, self.block_count = mode, len(block_dims)
        if mode == 'concat':
            self.proj = nn.Linear(sum(block_dims), out_dim)
        elif mode in ('max', 'attention'):
            self.projs = nn.ModuleList(nn.Linear(d, out_dim) for d in block_dims)
            if mode == 'attention':
                self.attn_vecs = nn.ParameterList(
                    nn.Parameter(torch.randn(out_dim)) for _ in block_dims)

    def forward(self, blocks):
        if self.mode == 'concat':
            return self.proj(torch.cat(blocks, 1))
        elif self.mode == 'max':
            stacked = torch.stack([p(b) for p, b in zip(self.projs, blocks)], 0)
            return torch.max(stacked, 0)[0]
        elif self.mode == 'attention':
            proj = [p(b) for p, b in zip(self.projs, blocks)]
            scores = [ (b * v).sum(-1) / math.sqrt(b.size(-1))
                       for b, v in zip(proj, self.attn_vecs) ]  # (D)
            w = torch.stack(scores, 1).softmax(1)               # [N, B]
            out = sum(w[:, i:i+1] * proj[i] for i in range(len(proj)))
            return out

class DJMGNN(nn.Module):
    def __init__(self,
                 in_dim, hidden_dim,
                 n_blocks=3, layers_per_block=6,
                 edge_attr_dim=0,
                 jk_mode='attention',
                 node_out_dim=1, graph_out_dim=1,
                 dropout=0.2,
                 pool_type='mean',
                 p_dropedge=0.1,
                 use_supernode=True,
                 use_rbf=True, rbf_K=32):
        super().__init__()
        self.p_dropedge, self.use_super, self.use_rbf, self.rbf_K = \
            p_dropedge, use_supernode, use_rbf, rbf_K

        # Build dense blocks
        self.blocks = nn.ModuleList()
        for b in range(n_blocks):
            self.blocks.append(
                DenseGNNBlock(
                    in_dim=hidden_dim if b else in_dim,
                    hidden_dim=hidden_dim,
                    n_layers=layers_per_block,
                    transition_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim + (rbf_K if use_rbf else 0)
                )
            )

        self.jk = JKAggregator([hidden_dim]*n_blocks, hidden_dim, mode=jk_mode)

        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, node_out_dim)
        )
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, graph_out_dim)
        )
        self.pool = {'mean': global_mean_pool,
                     'add':  global_add_pool,
                     'max':  global_max_pool
                    }.get(pool_type, global_mean_pool)

    # ---------- data‑level helpers -------------
    def add_supernode(self, x, edge_index, edge_attr, batch):
        """Add one virtual node per graph and connect it to all nodes."""
        if not self.use_super:                                # skip if disabled
            return x, edge_index, edge_attr, batch
        n = x.size(0)
        super_feat = x.new_zeros((batch.max().item()+1, x.size(1)))
        x = torch.cat([x, super_feat], 0)

        # build edges centre⟷node
        device = x.device
        row = torch.arange(n, device=device)
        col = batch                                      # destination super node id
        col = col + n                                    # shift indices after concat
        edge1 = torch.stack([row, col], 0)
        edge2 = torch.stack([col, row], 0)
        edge_index = torch.cat([edge_index, edge1, edge2], 1)

        if edge_attr is not None:
            super_e = edge_attr.new_zeros(edge1.size(1)*2, edge_attr.size(1))
            edge_attr = torch.cat([edge_attr, super_e], 0)

        batch = torch.cat([batch, torch.arange(super_feat.size(0), device=device)])
        return x, edge_index, edge_attr, batch

    def drop_edges(self, edge_index, edge_attr):
        if not self.training or self.p_dropedge == 0:          # inference: keep all
            return edge_index, edge_attr
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.p_dropedge
        return edge_index[:, mask], (edge_attr[mask] if edge_attr is not None else None)

    def forward(self, x, edge_index, edge_attr=None, batch=None, dist=None):
        """
        dist: optional tensor [num_edges, 1] of inter‑atomic distances
        """
        # RBF encode distances and concatenate to edge_attr  (H)
        if self.use_rbf and dist is not None:
            rbf = rbf_encode_dist(dist, K=self.rbf_K)          # [E, K]
            edge_attr = rbf if edge_attr is None else torch.cat([edge_attr, rbf], 1)

        # Add virtual super node  (G)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        x, edge_index, edge_attr, batch = self.add_supernode(x, edge_index, edge_attr, batch)

        # DropEdge regularisation  (F)
        edge_index, edge_attr = self.drop_edges(edge_index, edge_attr)

        # Backbone
        h, outs = x, []
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
            outs.append(h)
        h = self.jk(outs)                                      # node embeddings

        node_pred = self.node_head(h)

        graph_pred = self.graph_head(self.pool(h, batch))
        return {'node_pred': node_pred, 'graph_pred': graph_pred}
