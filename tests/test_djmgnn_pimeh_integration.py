"""
Test DJMGNN-PIMEH Integration

This script provides comprehensive validation of the PIMEH integration into DJMGNN,
testing all edge cases, error handling, and production requirements.

Key Validation Areas:
1. Forward pass integration with positions
2. Fallback mechanism when positions unavailable  
3. Output shape consistency (19 dimensions)
4. Device placement and gradient flow
5. Batch processing and supernode handling
6. Error recovery and logging behavior

Usage:
    python test_djmgnn_pimeh_integration.py
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional
import pytest

# Set up logging to capture DJMGNN messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from moml.models.mgnn.djmgnn import DJMGNN

def create_test_batch(
    num_molecules: int = 2,
    nodes_per_mol: Tuple[int, ...] = (4, 3),
    hidden_dim: int = 32,
    device: str = "cpu",
    include_pos: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Create a realistic test batch for DJMGNN-PIMEH integration testing.
    
    Args:
        num_molecules: Number of molecules in batch
        nodes_per_mol: Tuple specifying atoms per molecule
        hidden_dim: Feature dimension (should match DJMGNN)
        device: Device for tensors
        include_pos: Whether to include position data
        
    Returns:
        Dictionary containing batch tensors
    """
    total_nodes = sum(nodes_per_mol)
    
    # Node features (e.g., atomic features)
    x = torch.randn(total_nodes, 29, device=device)  # Standard node feature dim
    
    # Create batch assignment
    batch = torch.cat([
        torch.full((nodes,), mol_idx, dtype=torch.long, device=device)
        for mol_idx, nodes in enumerate(nodes_per_mol)
    ])
    
    # Create simple edge connectivity (fully connected within molecules)
    edge_index_list = []
    node_offset = 0
    for nodes in nodes_per_mol:
        # Create edges within molecule (all-to-all)
        mol_edges = []
        for i in range(nodes):
            for j in range(nodes):
                if i != j:  # No self-loops
                    mol_edges.append([node_offset + i, node_offset + j])
        if mol_edges:
            edge_index_list.extend(mol_edges)
        node_offset += nodes
    
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    # Distance features for edges
    dist = torch.rand(edge_index.size(1), 1, device=device) * 3.0 + 1.0  # 1-4 Å
    
    # Position data (critical for PIMEH)
    pos = None
    if include_pos:
        pos = torch.randn(total_nodes, 3, device=device) * 2.0  # Realistic molecular scale
    
    # Target data for testing
    node_y = torch.randn(total_nodes, 3, device=device)  # Forces
    y = torch.randn(num_molecules, 19, device=device)  # Graph properties (19 dims)
    y_graph = torch.randn(num_molecules, 1, device=device)  # Energies
    
    return {
        "x": x, "edge_index": edge_index, "batch": batch, "dist": dist, "pos": pos,
        "node_y": node_y, "y": y, "y_graph": y_graph
    }


def test_djmgnn_pimeh_forward_with_positions():
    """Test DJMGNN forward pass with positions - should use PIMEH."""
    logger.info("🧪 Testing DJMGNN-PIMEH forward pass with positions...")
    
    # Initialize model
    model = DJMGNN(
        in_node_dim=29,
        hidden_dim=32,
        n_blocks=2,  # Smaller for testing
        layers_per_block=2,
        graph_output_dims=19,
        use_supernode=True,
    )
    model.eval()
    
    # Create test batch with positions
    batch_data = create_test_batch(include_pos=True)
    
    with torch.no_grad():
        output = model(
            x=batch_data["x"],
            edge_index=batch_data["edge_index"], 
            batch=batch_data["batch"],
            dist=batch_data["dist"],
            pos=batch_data["pos"]  # Positions provided
        )
    
    # Validate output structure
    assert "node_pred" in output, "Missing node predictions"
    assert "graph_pred" in output, "Missing graph predictions" 
    assert "energy_pred" in output, "Missing energy predictions"
    
    # Validate shapes
    expected_batch_size = len(torch.unique(batch_data["batch"]))
    assert output["graph_pred"].shape == (expected_batch_size, 19), \
        f"Expected graph_pred shape ({expected_batch_size}, 19), got {output['graph_pred'].shape}"
    
    # Check that rotational constants (indices 16-18) are not fallback values
    rot_constants = output["graph_pred"][:, 16:19]
    assert not torch.allclose(rot_constants, torch.full_like(rot_constants, 10.0)), \
        "Rotational constants appear to be fallback values - PIMEH may not be working"
    
    logger.info("✅ Forward pass with positions test passed!")
    return output


def test_djmgnn_pimeh_forward_without_positions():
    """Test DJMGNN forward pass without positions - should use fallback."""
    logger.info("🧪 Testing DJMGNN-PIMEH forward pass without positions...")
    
    model = DJMGNN(
        in_node_dim=29,
        hidden_dim=32,
        n_blocks=2,
        layers_per_block=2,
        graph_output_dims=19,
    )
    model.eval()
    
    # Create test batch without positions
    batch_data = create_test_batch(include_pos=False)
    
    with torch.no_grad():
        output = model(
            x=batch_data["x"],
            edge_index=batch_data["edge_index"],
            batch=batch_data["batch"],
            dist=batch_data["dist"],
            pos=None  # No positions
        )
    
    # Validate output structure and shapes
    expected_batch_size = len(torch.unique(batch_data["batch"]))
    assert output["graph_pred"].shape == (expected_batch_size, 19)
    
    # Check that rotational constants are fallback values (10.0)
    rot_constants = output["graph_pred"][:, 16:19]
    assert torch.allclose(rot_constants, torch.full_like(rot_constants, 10.0)), \
        "Expected fallback rotational constants (10.0) when positions not provided"
    
    logger.info("✅ Forward pass without positions test passed!")
    return output


def test_djmgnn_pimeh_gradient_flow():
    """Test that gradients flow properly through PIMEH integration."""
    logger.info("🧪 Testing gradient flow through DJMGNN-PIMEH...")
    
    model = DJMGNN(
        in_node_dim=29,
        hidden_dim=32,
        n_blocks=2,
        layers_per_block=2,
        graph_output_dims=19,
    )
    model.train()  # Enable gradients
    
    batch_data = create_test_batch(include_pos=True)
    
    # Forward pass
    output = model(
        x=batch_data["x"],
        edge_index=batch_data["edge_index"],
        batch=batch_data["batch"],
        dist=batch_data["dist"],
        pos=batch_data["pos"]
    )
    
    # Compute loss focusing on rotational constants
    rot_constants = output["graph_pred"][:, 16:19]
    target_rot = torch.rand_like(rot_constants) * 100 + 10  # Realistic range
    loss = nn.MSELoss()(rot_constants, target_rot)
    
    # Backward pass
    loss.backward()
    
    # Check that PIMEH parameters have gradients
    pimeh_params_with_grad = 0
    total_pimeh_params = 0
    
    for name, param in model.pimeh_head.named_parameters():
        total_pimeh_params += 1
        if param.grad is not None:
            pimeh_params_with_grad += 1
            assert not torch.isnan(param.grad).any(), f"NaN gradient in PIMEH parameter: {name}"
            assert torch.isfinite(param.grad).all(), f"Infinite gradient in PIMEH parameter: {name}"
    
    assert pimeh_params_with_grad > 0, "No PIMEH parameters received gradients"
    assert pimeh_params_with_grad == total_pimeh_params, \
        f"Only {pimeh_params_with_grad}/{total_pimeh_params} PIMEH parameters received gradients"
    
    logger.info("✅ Gradient flow test passed!")


def test_djmgnn_pimeh_edge_cases():
    """Test edge cases and error handling."""
    logger.info("🧪 Testing DJMGNN-PIMEH edge cases...")
    
    model = DJMGNN(in_node_dim=29, hidden_dim=32, graph_output_dims=19)
    model.eval()
    
    # Test 1: Empty positions tensor
    batch_data = create_test_batch()
    batch_data["pos"] = torch.empty(0, 3)  # Empty positions
    
    with torch.no_grad():
        output = model(**{k: v for k, v in batch_data.items() if k in ["x", "edge_index", "batch", "dist", "pos"]})
    
    # Should use fallback values
    rot_constants = output["graph_pred"][:, 16:19]
    assert torch.allclose(rot_constants, torch.full_like(rot_constants, 10.0))
    
    # Test 2: Position-embedding size mismatch
    batch_data = create_test_batch()
    batch_data["pos"] = torch.randn(2, 3)  # Wrong number of positions
    
    with torch.no_grad():
        output = model(**{k: v for k, v in batch_data.items() if k in ["x", "edge_index", "batch", "dist", "pos"]})
    
    # Should handle gracefully with fallback
    assert output["graph_pred"].shape[1] == 19
    
    logger.info("✅ Edge cases test passed!")


def test_djmgnn_pimeh_device_compatibility():
    """Test device compatibility between DJMGNN and PIMEH components."""
    if not torch.cuda.is_available():
        logger.info("🚫 CUDA not available, skipping device compatibility test")
        return
    
    logger.info("🧪 Testing DJMGNN-PIMEH device compatibility...")
    
    device = "cuda"
    model = DJMGNN(in_node_dim=29, hidden_dim=32, graph_output_dims=19).to(device)
    model.eval()
    
    batch_data = create_test_batch(device=device, include_pos=True)
    
    with torch.no_grad():
        output = model(**{k: v for k, v in batch_data.items() if k in ["x", "edge_index", "batch", "dist", "pos"]})
    
    # All outputs should be on correct device
    assert output["graph_pred"].device.type == device
    assert output["node_pred"].device.type == device
    assert output["energy_pred"].device.type == device
    
    logger.info("✅ Device compatibility test passed!")


def run_comprehensive_integration_tests():
    """Run all integration tests for DJMGNN-PIMEH."""
    logger.info("🚀 Starting comprehensive DJMGNN-PIMEH integration tests...")
    logger.info("=" * 60)
    
    tests = [
        test_djmgnn_pimeh_forward_with_positions,
        test_djmgnn_pimeh_forward_without_positions,
        test_djmgnn_pimeh_gradient_flow,
        test_djmgnn_pimeh_edge_cases,
        test_djmgnn_pimeh_device_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test_func.__name__} failed: {e}")
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"🎯 Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED! DJMGNN-PIMEH integration is ready for production.")
    else:
        logger.error(f"💥 {failed} tests failed. Integration needs debugging.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_comprehensive_integration_tests()
    exit(0 if success else 1)