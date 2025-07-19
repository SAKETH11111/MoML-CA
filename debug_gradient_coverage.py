#!/usr/bin/env python3
"""
debug_gradient_coverage.py

Check actual gradient coverage in the joint MGNN model for research standards.
"""
import torch
import sys
sys.path.append('.')

from moml.models.mgnn.joint_mgnn import create_joint_mgnn, JointMGNN
from torch_geometric.data import Data

def check_gradient_coverage():
    """Check actual gradient coverage with hierarchical data."""
    
    # Test configurations from comprehensive test
    djmgnn_config = {
        'in_node_dim': 16,
        'hidden_dim': 32,
        'n_blocks': 2,
        'layers_per_block': 2,
        'in_edge_dim': 4,
        'jk_mode': 'attention',
        'node_output_dims': 3,
        'graph_output_dims': 19, # Corrected to match model
        'energy_output_dims': 1,
        'dropout': 0.1
    }
    
    hmgnn_config = {
        'scale_dims': [16, 16, 16],
        'hidden_dim': 32,
        'n_blocks': 2,
        'layers_per_block': 2,
        'jk_mode': "attention",
        'node_out_dim': 3,
        'graph_out_dim': 19,
        'cross_scale_exchange': True,
        'dropout': 0.1,
        'n_heads_cs': 4
    }
    
    joint_config = {
        'fusion_dim': 64,
        'alpha': 0.5
    }
    
    # Create joint model
    joint_model = create_joint_mgnn(
        djmgnn_config=djmgnn_config,
        hmgnn_config=hmgnn_config,
        joint_config=joint_config
    )
    
    # Create test data
    num_nodes = 10
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_attr = torch.randn(edge_index.shape[1], 4)
    test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Create hierarchical data with edge attributes and better connectivity
    hierarchical_data = []
    for scale in range(3):
        num_nodes_scale = max(3, num_nodes // (scale + 1))  # At least 3 nodes
        x_scale = torch.randn(num_nodes_scale, 16)
        
        # Create fully connected graph with self-loops to ensure connectivity
        edge_pairs = []
        for i in range(num_nodes_scale):
            for j in range(num_nodes_scale):
                edge_pairs.append([i, j])
        
        edge_index_scale = torch.tensor(edge_pairs, dtype=torch.long).T
        num_edges = edge_index_scale.size(1)
        edge_attr_scale = torch.randn(num_edges, 4)  # 4-dim edge features
        batch_scale = torch.zeros(num_nodes_scale, dtype=torch.long)
        
        hierarchical_data.append({
            'x': x_scale,
            'edge_index': edge_index_scale,
            'edge_attr': edge_attr_scale,
            'batch': batch_scale
        })
    
    # Create mappings
    mappings = [
        torch.arange(num_nodes, dtype=torch.long),
        torch.arange(max(1, num_nodes // 2), dtype=torch.long),
    ]
    cluster_counts = [
        torch.ones(num_nodes, dtype=torch.long),
        torch.ones(max(1, num_nodes // 2), dtype=torch.long),
    ]
    
    # Forward pass with hierarchical data
    outputs = joint_model(
        x=test_data.x,
        edge_index=test_data.edge_index,
        edge_attr=test_data.edge_attr,
        scale_data=hierarchical_data,
        maps=(mappings, cluster_counts),
        use_fusion=True
    )
    
    # Compute loss
    loss, _ = joint_model.compute_joint_loss(
        outputs,
        targets={
            'molecular_properties': torch.randn(1, 19),
            'forces': torch.randn(num_nodes, 3),
            'node': (torch.randn(num_nodes, 3), torch.ones(num_nodes, dtype=torch.bool)),
            'graph': (torch.randn(1, 19), torch.ones(1, dtype=torch.bool))
        }
    )
    
    # Backward pass
    loss.backward()
    
    # Check gradient coverage
    total_params = 0
    gradients_exist = 0
    no_grad_params = []
    
    for name, param in joint_model.named_parameters():
        total_params += 1
        if param.grad is not None and param.grad.abs().sum() > 1e-8:
            gradients_exist += 1
        else:
            no_grad_params.append(name)
    
    gradient_ratio = gradients_exist / total_params
    
    print(f"GRADIENT COVERAGE ANALYSIS")
    print(f"=" * 50)
    print(f"Total parameters: {total_params}")
    print(f"Parameters with gradients: {gradients_exist}")
    print(f"Gradient coverage: {gradient_ratio:.3f} ({gradient_ratio*100:.1f}%)")
    print(f"Parameters without gradients: {len(no_grad_params)}")
    
    if no_grad_params:
        print(f"\nParameters without gradients:")
        for param in no_grad_params[:10]:  # Show first 10
            print(f"  - {param}")
        if len(no_grad_params) > 10:
            print(f"  ... and {len(no_grad_params)-10} more")
    
    # Research recommendations
    print(f"\nRESEARCH STANDARDS:")
    print(f"✓ Production (80%): {'PASS' if gradient_ratio >= 0.80 else 'FAIL'}")
    print(f"✓ Research (90%): {'PASS' if gradient_ratio >= 0.90 else 'FAIL'}")  
    print(f"✓ Rigorous Research (95%): {'PASS' if gradient_ratio >= 0.95 else 'FAIL'}")
    print(f"✓ Cutting-edge (98%): {'PASS' if gradient_ratio >= 0.98 else 'FAIL'}")
    
    return gradient_ratio

if __name__ == "__main__":
    coverage = check_gradient_coverage()
    print(f"\nFinal gradient coverage: {coverage:.3f}")