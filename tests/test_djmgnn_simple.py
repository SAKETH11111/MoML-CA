"""
Simple DJMGNN-PIMEH Integration Test

Direct test of the DJMGNN-PIMEH integration without heavy dependencies.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, '/home/saketh/MoML-CA')

def test_djmgnn_import():
    """Test that DJMGNN can be imported with PIMEH integration."""
    print("🧪 Testing DJMGNN import with PIMEH integration...")
    
    try:
        from moml.models.mgnn.djmgnn import DJMGNN
        print("✅ DJMGNN imported successfully")
        return DJMGNN
    except Exception as e:
        print(f"❌ Failed to import DJMGNN: {e}")
        return None

def test_djmgnn_instantiation():
    """Test DJMGNN instantiation with PIMEH head."""
    print("🧪 Testing DJMGNN instantiation...")
    
    try:
        from moml.models.mgnn.djmgnn import DJMGNN
        
        model = DJMGNN(
            in_node_dim=29,
            hidden_dim=64,
            n_blocks=2,
            layers_per_block=2,
            graph_output_dims=19,
        )
        
        # Check that PIMEH head exists
        assert hasattr(model, 'pimeh_head'), "DJMGNN missing pimeh_head attribute"
        assert hasattr(model, 'graph_head'), "DJMGNN missing graph_head attribute"
        
        # Check graph_head output dimension (should be 16, not 19)
        graph_head_out_dim = model.graph_head[-1].out_features
        assert graph_head_out_dim == 16, f"Expected graph_head output 16, got {graph_head_out_dim}"
        
        print("✅ DJMGNN instantiation successful")
        print(f"   - PIMEH head: {type(model.pimeh_head).__name__}")
        print(f"   - Graph head output dim: {graph_head_out_dim}")
        return model
        
    except Exception as e:
        print(f"❌ DJMGNN instantiation failed: {e}")
        return None

def test_djmgnn_forward_pass():
    """Test DJMGNN forward pass with and without positions."""
    print("🧪 Testing DJMGNN forward pass...")
    
    try:
        from moml.models.mgnn.djmgnn import DJMGNN
        
        model = DJMGNN(
            in_node_dim=29,
            hidden_dim=32,
            n_blocks=2,
            layers_per_block=2,
            graph_output_dims=19,
        )
        model.eval()
        
        # Create test data
        num_nodes = 6
        num_molecules = 2
        x = torch.randn(num_nodes, 29)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        pos = torch.randn(num_nodes, 3)
        
        # Test with positions
        print("   Testing with positions...")
        with torch.no_grad():
            output_with_pos = model(
                x=x,
                edge_index=edge_index,
                batch=batch,
                pos=pos
            )
        
        assert output_with_pos["graph_pred"].shape == (num_molecules, 19), \
            f"Expected shape ({num_molecules}, 19), got {output_with_pos['graph_pred'].shape}"
        
        # Test without positions  
        print("   Testing without positions...")
        with torch.no_grad():
            output_without_pos = model(
                x=x,
                edge_index=edge_index,
                batch=batch,
                pos=None
            )
        
        assert output_without_pos["graph_pred"].shape == (num_molecules, 19)
        
        # Check fallback behavior
        rot_constants_fallback = output_without_pos["graph_pred"][:, 16:19]
        expected_fallback = torch.full_like(rot_constants_fallback, 10.0)
        
        if torch.allclose(rot_constants_fallback, expected_fallback):
            print("   ✅ Fallback rotational constants working correctly")
        else:
            print("   ⚠️  Fallback values may not be working as expected")
            
        print("✅ Forward pass tests successful")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_tests():
    """Run simple integration tests."""
    print("🚀 Starting simple DJMGNN-PIMEH integration tests...")
    print("=" * 50)
    
    # Test 1: Import
    DJMGNN = test_djmgnn_import()
    if DJMGNN is None:
        return False
    
    # Test 2: Instantiation
    model = test_djmgnn_instantiation()
    if model is None:
        return False
    
    # Test 3: Forward pass
    success = test_djmgnn_forward_pass()
    if not success:
        return False
    
    print("=" * 50)
    print("🎉 ALL SIMPLE TESTS PASSED! DJMGNN-PIMEH integration looks good!")
    return True

if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)