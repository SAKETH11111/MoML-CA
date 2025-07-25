#!/usr/bin/env python3
"""
Test script to calculate parameter count increase from capacity enhancement.

This script compares parameter counts between:
- Current configuration: hidden_dim=128, n_blocks=3 vs 4
- New configuration: hidden_dim=160, n_blocks=4

Usage:
    python test_capacity_increase.py
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/saketh/MoML-CA')

from moml.models.mgnn.djmgnn import DJMGNN


def count_parameters(model):
    """Count trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Break down by component
    component_counts = {}
    
    # PIMEH parameters
    if hasattr(model, 'pimeh_head'):
        pimeh_params = sum(p.numel() for p in model.pimeh_head.parameters() if p.requires_grad)
        component_counts['pimeh'] = pimeh_params
    else:
        component_counts['pimeh'] = 0
    
    # Base model parameters (everything except PIMEH)
    base_params = total_params - component_counts['pimeh']
    component_counts['base'] = base_params
    component_counts['total'] = total_params
    
    return component_counts


def create_model_config(hidden_dim, n_blocks):
    """Create DJMGNN model with specified configuration."""
    return DJMGNN(
        in_node_dim=29,          # Standard molecular features
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        layers_per_block=2,      # Keep consistent with current setup
        graph_output_dims=19,    # QM9 properties
        node_output_dims=3,      # Forces
        energy_output_dims=1,    # Energy
        dropout=0.2,
        jk_mode="attention",
        use_supernode=True,
        use_rbf=True,
        rbf_K=32
    )


def test_parameter_scaling():
    """Test parameter scaling with different configurations."""
    print("🧮 Parameter Count Analysis for DJMGNN Capacity Increase")
    print("=" * 70)
    
    configs = [
        ("Current (template)", 128, 3),
        ("Current (joint/code)", 128, 4), 
        ("New (target)", 160, 4),
    ]
    
    results = {}
    
    for name, hidden_dim, n_blocks in configs:
        print(f"\n📊 Configuration: {name}")
        print(f"   hidden_dim: {hidden_dim}, n_blocks: {n_blocks}")
        
        # Create model
        model = create_model_config(hidden_dim, n_blocks)
        
        # Count parameters
        counts = count_parameters(model)
        results[name] = counts
        
        # Display results
        print(f"   Total parameters: {counts['total']:,}")
        print(f"   Base DJMGNN: {counts['base']:,}")
        print(f"   PIMEH Head: {counts['pimeh']:,}")
        print(f"   PIMEH %: {100 * counts['pimeh'] / counts['total']:.1f}%")
    
    # Calculate increases
    print(f"\n📈 Parameter Increases:")
    print("=" * 70)
    
    # From template config to new config
    if "Current (template)" in results and "New (target)" in results:
        current = results["Current (template)"]
        new = results["New (target)"]
        
        total_increase = new['total'] - current['total']
        percent_increase = 100 * total_increase / current['total']
        
        print(f"From template config (128/3) to new config (160/4):")
        print(f"   Total increase: +{total_increase:,} parameters (+{percent_increase:.1f}%)")
        print(f"   Base DJMGNN: +{new['base'] - current['base']:,}")
        print(f"   PIMEH Head: +{new['pimeh'] - current['pimeh']:,}")
    
    # From joint config to new config (most likely current state)
    if "Current (joint/code)" in results and "New (target)" in results:
        current = results["Current (joint/code)"]
        new = results["New (target)"]
        
        total_increase = new['total'] - current['total']
        percent_increase = 100 * total_increase / current['total']
        
        print(f"\nFrom joint config (128/4) to new config (160/4):")
        print(f"   Total increase: +{total_increase:,} parameters (+{percent_increase:.1f}%)")
        print(f"   Base DJMGNN: +{new['base'] - current['base']:,}")
        print(f"   PIMEH Head: +{new['pimeh'] - current['pimeh']:,}")
    
    # Memory estimation
    print(f"\n💾 Memory Impact Estimation:")
    print("=" * 70)
    
    if "New (target)" in results:
        new_params = results["New (target)"]["total"]
        
        # Estimate memory usage (rough approximation)
        # Parameters: 4 bytes per float32 parameter
        # Gradients: 4 bytes per parameter (same size as parameters)
        # Optimizer state (AdamW): ~8 bytes per parameter (momentum + variance)
        # Forward activations: Depends on batch size, roughly 2-4x parameters
        
        param_memory_mb = new_params * 4 / (1024 * 1024)  # MB
        grad_memory_mb = new_params * 4 / (1024 * 1024)   # MB  
        optimizer_memory_mb = new_params * 8 / (1024 * 1024)  # MB
        activation_memory_mb = new_params * 3 / (1024 * 1024)  # MB (estimate)
        
        total_memory_mb = param_memory_mb + grad_memory_mb + optimizer_memory_mb + activation_memory_mb
        
        print(f"   Parameters: {param_memory_mb:.1f} MB")
        print(f"   Gradients: {grad_memory_mb:.1f} MB")
        print(f"   Optimizer state: {optimizer_memory_mb:.1f} MB")
        print(f"   Activations (est): {activation_memory_mb:.1f} MB")
        print(f"   Total (est): {total_memory_mb:.1f} MB ({total_memory_mb/1024:.2f} GB)")
        
        if total_memory_mb < 2000:  # 2GB
            print("   ✅ Memory usage appears feasible for most GPUs")
        elif total_memory_mb < 8000:  # 8GB
            print("   ⚠️  Memory usage moderate - should work on 8GB+ GPUs")
        else:
            print("   ❌ High memory usage - may require 16GB+ GPU")
    
    print(f"\n✅ Parameter analysis complete!")
    return results


def test_dimension_compatibility():
    """Test that all dimensions are compatible in the new configuration."""
    print(f"\n🔧 Testing Dimension Compatibility")
    print("=" * 70)
    
    hidden_dim = 160
    n_blocks = 4
    
    try:
        # Create model
        model = create_model_config(hidden_dim, n_blocks)
        
        # Create mock batch
        batch_size = 2
        num_nodes = 8
        
        x = torch.randn(num_nodes, 29)  # Node features
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                                   [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        dist = torch.rand(edge_index.size(1), 1) * 3.0 + 1.0
        pos = torch.randn(num_nodes, 3)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(
                x=x,
                edge_index=edge_index,
                batch=batch,
                dist=dist,
                pos=pos
            )
        
        # Validate output shapes
        expected_shapes = {
            'node_pred': (num_nodes, 3),
            'graph_pred': (batch_size, 19),
            'energy_pred': (batch_size, 1)
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = output[key].shape
            assert actual_shape == expected_shape, f"{key}: expected {expected_shape}, got {actual_shape}"
            print(f"   ✅ {key}: {actual_shape}")
        
        # Test PIMEH specifically
        if hasattr(model, 'pimeh_head'):
            # Test PIMEH input/output
            mock_h = torch.randn(num_nodes, hidden_dim)
            pimeh_output = model.pimeh_head(mock_h, pos, batch)
            expected_pimeh_shape = (batch_size, 3)
            actual_pimeh_shape = pimeh_output.shape
            assert actual_pimeh_shape == expected_pimeh_shape, f"PIMEH: expected {expected_pimeh_shape}, got {actual_pimeh_shape}"
            print(f"   ✅ PIMEH output: {actual_pimeh_shape}")
        
        print(f"   ✅ All dimensions compatible!")
        return True
        
    except Exception as e:
        print(f"   ❌ Dimension compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 DJMGNN Capacity Increase Analysis")
    print("="*70)
    
    # Run parameter analysis
    results = test_parameter_scaling()
    
    # Run dimension compatibility test
    compatibility_success = test_dimension_compatibility()
    
    print(f"\n🎯 Summary:")
    print("="*70)
    print("📋 Changes needed:")
    print("   1. hidden_dim: 128 → 160 (+25%)")
    print("   2. n_blocks: 3 → 4 (or keep at 4 if already there)")
    print("   3. PIMEH automatically scales with hidden_dim")
    
    if compatibility_success:
        print("✅ All dimension compatibility tests passed")
        print("🚀 Ready to implement capacity increase!")
    else:
        print("❌ Dimension compatibility issues detected")
        print("⚠️  Fix issues before proceeding")
    
    print(f"\n💡 Next steps:")
    print("   1. Update configuration files (training_config.template.yaml, joint_training.yaml)")
    print("   2. Update training script defaults if needed")
    print("   3. Test training with new configuration")
    print("   4. Monitor memory usage during training")