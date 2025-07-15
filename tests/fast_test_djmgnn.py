import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.models.mgnn.djmgnn import DJMGNN

def test_djmgnn_output_shapes():
    model = DJMGNN(in_node_dim=29, hidden_dim=64, n_blocks=2, layers_per_block=2)
    
    # Create random data
    num_nodes = 10
    x = torch.randn(num_nodes, 29)
    edge_index = torch.randint(0, num_nodes, (2, 2))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Forward pass
    out = model(x=x, edge_index=edge_index, batch=batch)
    
    # Assert output shapes
    assert out['graph_pred'].shape == (1, 19)
    assert out['node_pred'].shape == (10, 3)
    
    print("Fast unit test passed!")

if __name__ == "__main__":
    test_djmgnn_output_shapes()