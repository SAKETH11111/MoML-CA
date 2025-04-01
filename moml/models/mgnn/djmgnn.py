import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.nn import global_mean_pool

class GraphConvLayer(nn.Module):
    """
    A graph convolution layer based on NNConv, which allows the model
    to learn from bond (edge) attributes in addition to node features.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_attr_dim: int):
        """
        Args:
            in_channels: Dimensionality of node features coming in
            out_channels: Dimensionality of output node features
            edge_attr_dim: Dimensionality of the bond (edge) feature vector
        """
        super().__init__()
        
        # MLP that maps each edge_attr -> a [in_channels * out_channels] weight matrix
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, in_channels * out_channels),
            nn.ReLU()
        )
        
        # NNConv applies a learned linear transform (via edge_mlp) to each neighbor's features
        self.conv = NNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            nn=self.edge_mlp,
            aggr='add'  # or 'mean', 'max', depending on your preference
        )
        
        # Optional batch norm on output features
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of NNConv.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge feature matrix [num_edges, edge_attr_dim]
        
        Returns:
            Updated node feature matrix [num_nodes, out_channels]
        """
        # Perform NNConv aggregation
        x = self.conv(x, edge_index, edge_attr)
        
        # Apply batch norm (requires [batch_size, num_features])
        x = self.bn(x)
        
        # Nonlinear activation
        x = F.relu(x)
        
        return x

        

class DenseGNNBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        transition_dim: int = None,
        edge_attr_dim: int = 0
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.current_dim = in_dim  # total dimension so far
        
        # Build n_layers, each sees concatenated outputs from all prior layers
        for _ in range(n_layers):
            self.layers.append(
                GraphConvLayer(
                    in_channels=self.current_dim,
                    out_channels=hidden_dim,
                    edge_attr_dim=edge_attr_dim
                )
            )
            self.current_dim += hidden_dim
        
        out_dim = self.current_dim
        final_dim = transition_dim or hidden_dim
        self.transition = nn.Linear(out_dim, final_dim)
        self.bn = nn.BatchNorm1d(final_dim)

    def forward(self, x, edge_index, edge_attr=None):
        out_list = [x]
        for layer in self.layers:
            concat_x = torch.cat(out_list, dim=1)
            h = layer(concat_x, edge_index, edge_attr)
            out_list.append(h)
        dense_out = torch.cat(out_list, dim=1)
        out = self.transition(dense_out)
        out = self.bn(out)
        out = F.relu(out)
        return out


class JKAggregator(nn.Module):
    def __init__(self, block_dims, out_dim, mode='concat'):
        super().__init__()
        self.mode = mode
        self.block_count = len(block_dims)

        if mode == 'concat':
            in_dim = sum(block_dims)
            self.proj = nn.Linear(in_dim, out_dim)
        elif mode == 'max':
            self.projs = nn.ModuleList(
                nn.Linear(dim, out_dim) for dim in block_dims
            )
        elif mode == 'attention':
            self.attn_params = nn.ParameterList([
                nn.Parameter(torch.randn(1, dim)) for dim in block_dims
            ])
            self.projs = nn.ModuleList([
                nn.Linear(dim, out_dim) for dim in block_dims
            ])
        elif mode == 'lstm':
            self.hidden_size = out_dim
            max_dim = max(block_dims)
            self.projs = nn.ModuleList([
                nn.Linear(dim, max_dim) for dim in block_dims
            ])
            self.lstm = nn.LSTM(input_size=max_dim, hidden_size=out_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown JK mode: {mode}")

    def forward(self, block_outputs):
        """
        block_outputs: list of [batch_size, block_dims[i]] node embeddings from each block
        """
        if self.mode == 'concat':
            x_cat = torch.cat(block_outputs, dim=1)
            return self.proj(x_cat)
        elif self.mode == 'max':
            # project each block output
            ps = [self.projs[i](h) for i, h in enumerate(block_outputs)]
            stack = torch.stack(ps, dim=0)  # [num_blocks, batch_size, out_dim]
            out, _ = torch.max(stack, dim=0) 
            return out
        elif self.mode == 'attention':
            attn_scores = []
            for i, h in enumerate(block_outputs):
                score = torch.matmul(h, self.attn_params[i].t())  # [batch_size, 1]
                attn_scores.append(score)
            scores_cat = torch.cat(attn_scores, dim=1)           # [batch_size, num_blocks]
            attn_weights = F.softmax(scores_cat, dim=1)          # [batch_size, num_blocks]
            # project each block output
            proj_h = [self.projs[i](block_outputs[i]) for i in range(self.block_count)]
            weighted_sum = 0
            for i, p in enumerate(proj_h):
                w = attn_weights[:, i].unsqueeze(1)
                weighted_sum = weighted_sum + w * p
            return weighted_sum
        elif self.mode == 'lstm':
            seq_list = []
            for i, h in enumerate(block_outputs):
                # project each block output to the same dimension
                p = self.projs[i](h)
                seq_list.append(p)
            # shape = [batch_size, num_blocks, max_dim]
            packed = torch.stack(seq_list, dim=1)
            _, (h_n, _) = self.lstm(packed)
            out = h_n.squeeze(0)  # [batch_size, out_dim]
            return out

        
class DJMGNN(nn.Module):
    """
    Dense + Jumping Knowledge GNN with multi-task heads:
      - A node-level head for force-field parameters (e.g., partial charges).
      - A graph-level head for macroscopic property (e.g., adsorption affinity).
      - Additional heads if you like.

    The forward pass returns a dictionary of outputs:
      {
         'node_pred': [num_nodes, node_out_dim],  # optional
         'graph_pred': [batch_size, graph_out_dim], # optional
         ... 
      }
    so you can apply partial-labeled loss.
    """
    def __init__(
        self,
        in_dim: int,            # Node feature dimension
        hidden_dim: int,        # Growth dimension for each block
        n_blocks: int = 3,      # number of DenseGNNBlocks
        layers_per_block: int = 8,
        edge_attr_dim: int = 0,
        jk_mode: str = 'attention',
        # Node-level output dimension for force-field, e.g. partial charges or bond constants
        node_out_dim: int = 1,
        # Graph-level output dimension for a property (adsorption, etc.)
        graph_out_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Build Dense blocks 
        self.blocks = nn.ModuleList()
        block_dims = []

        # First block
        self.blocks.append(
            DenseGNNBlock(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                n_layers=layers_per_block,
                transition_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim
            )
        )
        block_dims.append(hidden_dim)

        # Additional blocks
        for _ in range(n_blocks - 1):
            self.blocks.append(
                DenseGNNBlock(
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    n_layers=layers_per_block,
                    transition_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim
                )
            )
            block_dims.append(hidden_dim)

        # JK aggregator (node-level)
        # merges outputs from each block into final node-level embedding
        self.jk_aggregator = JKAggregator(
            block_dims=block_dims,
            out_dim=hidden_dim,
            mode=jk_mode
        )

        # Multi-Task Heads
        # Node-level head: e.g. partial charges
        # We'll apply it directly to the node embeddings after JK aggregator
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, node_out_dim)
        )
        
        # Graph-level head: e.g. macroscopic property
        # We'll pool node embeddings to get a graph embedding, then do MLP
        self.graph_pool = global_mean_pool  # or an attention pool
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, graph_out_dim)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Return a dictionary of node-level and graph-level predictions:
          {
            'node_pred': Tensor [num_nodes, node_out_dim],
            'graph_pred': Tensor [batch_size, graph_out_dim]
          }
        Some tasks may want only one of these.
        """
        block_outputs = []
        h = x

        # Pass through each Dense block
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
            block_outputs.append(h)
        
        # Jumping Knowledge aggregator merges the node embeddings from each block
        jk_node_features = self.jk_aggregator(block_outputs)  # [num_nodes, hidden_dim]

        # A) Node-level prediction (e.g. partial charges)
        node_pred = self.node_head(jk_node_features)  # shape [num_nodes, node_out_dim]

        # B) Graph-level prediction (pool first, then MLP)
        if batch is None:
            # single-graph scenario: average all node embeddings
            graph_embed = jk_node_features.mean(dim=0, keepdim=True)
        else:
            # multiple graphs in batch
            graph_embed = self.graph_pool(jk_node_features, batch)
        graph_pred = self.graph_head(graph_embed)

        return {
            'node_pred': node_pred,     # [num_nodes, node_out_dim]
            'graph_pred': graph_pred    # [batch_size, graph_out_dim]
        } 