#!/usr/bin/env python3
"""Debug gradient flow in JointMGNN to identify which parameters lack gradients."""

import torch
from moml.models.mgnn.joint_mgnn import JointMGNN

def debug_gradient_flow():
    """Debug which parameters are getting gradients and which aren't."""
    
    # Create model (same as test)
    djmgnn_config = {
        'in_node_dim': 16,
        'hidden_dim': 32,
        'n_blocks': 2,
        'node_output_dims': 3,
        'graph_output_dims': 5,
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
    
    model = JointMGNN(
        djmgnn_config=djmgnn_config,
        hmgnn_config=hmgnn_config,
        fusion_dim=64,
        n_fusion_heads=4
    )
    
    # Create dummy data
    num_nodes = 10
    x = torch.randn(num_nodes, 16, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    edge_attr = torch.randn(20, 8)
    batch = None
    
    # Forward pass with fusion
    outputs = model(x, edge_index, edge_attr, batch=batch, use_fusion=True)
    
    # Compute loss
    loss = torch.tensor(0.0, requires_grad=True)
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            loss = loss + value.sum()
    
    print(f"Total loss: {loss}")
    print(f"Loss requires grad: {loss.requires_grad}")
    
    # Backward pass
    if loss.requires_grad:
        loss.backward()
    else:
        print("Loss does not require gradients!")
    
    # Check which parameters have gradients
    total_params = 0
    params_with_grad = 0
    
    print("\nParameter gradient analysis:")
    print("=" * 60)
    
    for name, param in model.named_parameters():
        total_params += 1
        has_grad = param.grad is not None
        if has_grad:
            params_with_grad += 1
        print(f"{name:<50} | Grad: {has_grad} | Shape: {param.shape}")
    
    print("=" * 60)
    print(f"Total parameters: {total_params}")
    print(f"Parameters with gradients: {params_with_grad}")
    print(f"Gradient ratio: {params_with_grad / total_params:.1%}")
    
    # Group by component
    component_stats = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0]  # Get first part (djmgnn, hmgnn, fusion_layer, etc.)
        if component not in component_stats:
            component_stats[component] = {'total': 0, 'with_grad': 0}
        component_stats[component]['total'] += 1
        if param.grad is not None:
            component_stats[component]['with_grad'] += 1
    
    print("\nGradient analysis by component:")
    print("=" * 60)
    for component, stats in component_stats.items():
        ratio = stats['with_grad'] / stats['total']
        print(f"{component:<20} | {stats['with_grad']}/{stats['total']} ({ratio:.1%})")


if __name__ == "__main__":
    debug_gradient_flow()