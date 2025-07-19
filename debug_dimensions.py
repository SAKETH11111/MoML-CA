#!/usr/bin/env python3

import torch
from moml.models.mgnn.djmgnn import DJMGNN
from moml.models.mgnn.hmgnn import HMGNN

# Test configurations from the test suite
djmgnn_config = {
    'in_node_dim': 16,
    'hidden_dim': 32,
    'n_blocks': 2,
    'layers_per_block': 2,
    'in_edge_dim': 0,
    'jk_mode': 'attention',
    'node_output_dims': 3,
    'graph_output_dims': 5,
    'energy_output_dims': 1,
    'dropout': 0.1
}

hmgnn_config = {
    'scale_dims': [16, 16, 16],
    'hidden_dim': 32,
    'n_blocks': 2,
    'layers_per_block': 2,
    'jk_mode': 'attention',
    'node_out_dim': 3,
    'graph_out_dim': 5,
    'cross_scale_exchange': True,
    'dropout': 0.1,
    'n_heads_cs': 4
}

# Create test data
num_nodes = 10
node_dim = 16
x = torch.randn(num_nodes, node_dim)
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
batch = torch.zeros(num_nodes, dtype=torch.long)

print("=== Testing DJMGNN ===")
djmgnn = DJMGNN(**djmgnn_config)
dj_out = djmgnn(x=x, edge_index=edge_index, batch=batch)
print(f"DJMGNN output keys: {dj_out.keys()}")
print(f"DJMGNN node_pred shape: {dj_out['node_pred'].shape}")
print(f"DJMGNN graph_pred shape: {dj_out['graph_pred'].shape}")

print("\n=== Testing HMGNN ===")
# Create simple scale data for HMGNN
scale_data = [{
    'x': x,
    'edge_index': edge_index,
    'edge_attr': None,
    'batch': batch
}]

hmgnn = HMGNN(**hmgnn_config)
hm_out = hmgnn(scale_data=scale_data, maps=None, edge_pairs_cs=None)
print(f"HMGNN output keys: {hm_out.keys()}")
print(f"HMGNN node_pred shape: {hm_out['node_pred'].shape if hm_out['node_pred'] is not None else 'None'}")
print(f"HMGNN graph_pred shape: {hm_out['graph_pred'].shape}")