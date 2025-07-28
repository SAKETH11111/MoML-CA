#!/usr/bin/env python3
"""
Test script for Project Apollo 4-Phase Curriculum Manager implementation.
Verifies the new sequential training schedule and freezing logic.
"""

import sys
import torch
import torch.optim as optim

# Add the project root to the path
sys.path.insert(0, '/home/saketh/MoML-CA')

try:
    from scripts.train_alternating_optimized import SimpleCurriculumManager, PHASE_0_END_STEP, PHASE_1_END_STEP, PHASE_2_END_STEP
    from moml.models.mgnn.djmgnn import DJMGNN
    print("✅ Successfully imported Apollo curriculum components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_phase_constants():
    """Test that the new 4-phase constants are correctly defined."""
    print("\n🔧 Testing Phase Constants...")
    
    # Verify the constants match Project Apollo specification
    assert PHASE_0_END_STEP == 8000, f"Expected PHASE_0_END_STEP=8000, got {PHASE_0_END_STEP}"
    assert PHASE_1_END_STEP == 10000, f"Expected PHASE_1_END_STEP=10000, got {PHASE_1_END_STEP}"
    assert PHASE_2_END_STEP == 10200, f"Expected PHASE_2_END_STEP=10200, got {PHASE_2_END_STEP}"
    
    print("✅ All phase constants correctly defined")
    return True

def test_phase_logic():
    """Test the get_current_phase logic for the 4-phase schedule."""
    print("\n📋 Testing Phase Logic...")
    
    # Create a dummy model for testing
    model = DJMGNN(
        in_node_dim=10,
        hidden_dim=64,
        n_blocks=2,
        layers_per_block=2
    )
    
    manager = SimpleCurriculumManager(model)
    
    # Test phase boundaries
    test_cases = [
        (0, 0),      # Phase 0: Backbone pre-training
        (4000, 0),   # Phase 0: Mid backbone training
        (7999, 0),   # Phase 0: End of backbone training
        (8000, 1),   # Phase 1: Start PIMEH adaptation
        (9000, 1),   # Phase 1: Mid PIMEH adaptation
        (9999, 1),   # Phase 1: End PIMEH adaptation
        (10000, 2),  # Phase 2: Start stability check
        (10100, 2),  # Phase 2: Mid stability check
        (10199, 2),  # Phase 2: End stability check
        (10200, 3),  # Phase 3: Start joint polishing
        (15000, 3),  # Phase 3: Mid joint polishing
        (20000, 3),  # Phase 3: Late joint polishing
    ]
    
    for step, expected_phase in test_cases:
        actual_phase = manager.get_current_phase(step)
        assert actual_phase == expected_phase, f"Step {step}: expected phase {expected_phase}, got {actual_phase}"
    
    print("✅ All phase logic tests passed")
    return True

def test_loss_weights():
    """Test the new 4-phase loss weight configuration."""
    print("\n⚖️ Testing Loss Weights...")
    
    model = DJMGNN(in_node_dim=10, hidden_dim=64, n_blocks=2, layers_per_block=2)
    manager = SimpleCurriculumManager(model)
    
    # Test Phase 0: Backbone pre-training (no physics, others only)
    weights_0 = manager.get_loss_weights(4000)  # Phase 0
    expected_0 = {'physics_loss': 0.0, 'node_loss': 1.0, 'graph_loss': 1.0, 'energy_loss': 1.0}
    assert weights_0 == expected_0, f"Phase 0 weights incorrect: {weights_0}"
    
    # Test Phase 1: PIMEH adaptation (physics only, no others)
    weights_1 = manager.get_loss_weights(9000)  # Phase 1
    expected_1 = {'physics_loss': 1.0, 'node_loss': 0.0, 'graph_loss': 0.0, 'energy_loss': 0.0}
    assert weights_1 == expected_1, f"Phase 1 weights incorrect: {weights_1}"
    
    # Test Phase 2: Stability check (inference only - all zeros)
    weights_2 = manager.get_loss_weights(10100)  # Phase 2
    expected_2 = {'physics_loss': 0.0, 'node_loss': 0.0, 'graph_loss': 0.0, 'energy_loss': 0.0}
    assert weights_2 == expected_2, f"Phase 2 weights incorrect: {weights_2}"
    
    # Test Phase 3: Joint polishing (all losses balanced)
    weights_3 = manager.get_loss_weights(15000)  # Phase 3
    expected_3 = {'physics_loss': 1.0, 'node_loss': 1.0, 'graph_loss': 1.0, 'energy_loss': 1.0}
    assert weights_3 == expected_3, f"Phase 3 weights incorrect: {weights_3}"
    
    print("✅ All loss weight tests passed")
    return True

def test_freezing_methods():
    """Test the new freezing methods for the 4-phase schedule."""
    print("\n🧊 Testing Freezing Methods...")
    
    # Create model with PIMEH adapter
    model = DJMGNN(in_node_dim=10, hidden_dim=64, n_blocks=2, layers_per_block=2)
    manager = SimpleCurriculumManager(model)
    
    # Test Phase 0: freeze_pimeh_and_adapter
    frozen_count, active_count = manager.freeze_pimeh_and_adapter()
    
    # Check that PIMEH and adapter parameters are frozen
    pimeh_frozen = all(not p.requires_grad for name, p in model.named_parameters() 
                      if name.startswith('pimeh_head') or name.startswith('pimeh_adapter'))
    assert pimeh_frozen, "PIMEH and adapter should be frozen in Phase 0"
    
    # Check that other parameters are active
    other_active = all(p.requires_grad for name, p in model.named_parameters() 
                      if not (name.startswith('pimeh_head') or name.startswith('pimeh_adapter')))
    assert other_active, "Non-PIMEH parameters should be active in Phase 0"
    
    # Test Phase 1: freeze_backbone
    frozen_count, active_count = manager.freeze_backbone()
    
    # Check that PIMEH and adapter parameters are active
    pimeh_active = all(p.requires_grad for name, p in model.named_parameters() 
                      if name.startswith('pimeh_head') or name.startswith('pimeh_adapter'))
    assert pimeh_active, "PIMEH and adapter should be active in Phase 1"
    
    # Check that backbone parameters are frozen
    backbone_frozen = all(not p.requires_grad for name, p in model.named_parameters() 
                         if not (name.startswith('pimeh_head') or name.startswith('pimeh_adapter')))
    assert backbone_frozen, "Backbone parameters should be frozen in Phase 1"
    
    # Test Phase 2: freeze_all
    frozen_count, active_count = manager.freeze_all()
    
    # Check that all parameters are frozen
    all_frozen = all(not p.requires_grad for p in model.parameters())
    assert all_frozen, "All parameters should be frozen in Phase 2"
    assert active_count == 0, "No parameters should be active in Phase 2"
    
    # Test Phase 3: unfreeze_all
    frozen_count, active_count = manager.unfreeze_all()
    
    # Check that all parameters are active
    all_active = all(p.requires_grad for p in model.parameters())
    assert all_active, "All parameters should be active in Phase 3"
    assert frozen_count == 0, "No parameters should be frozen in Phase 3"
    
    print("✅ All freezing method tests passed")
    return True

def test_gradnorm_control():
    """Test GradNorm skip logic for the new schedule."""
    print("\n🎯 Testing GradNorm Control...")
    
    model = DJMGNN(in_node_dim=10, hidden_dim=64, n_blocks=2, layers_per_block=2)
    manager = SimpleCurriculumManager(model)
    
    # Test that GradNorm is skipped in phases 0, 1, 2
    assert manager.should_skip_gradnorm(4000), "Should skip GradNorm in Phase 0"
    assert manager.should_skip_gradnorm(9000), "Should skip GradNorm in Phase 1"
    assert manager.should_skip_gradnorm(10100), "Should skip GradNorm in Phase 2"
    
    # Test that GradNorm is used in phase 3
    assert not manager.should_skip_gradnorm(15000), "Should use GradNorm in Phase 3"
    
    # Test inference-only phase detection
    assert not manager.is_inference_only_phase(4000), "Phase 0 is not inference-only"
    assert not manager.is_inference_only_phase(9000), "Phase 1 is not inference-only"
    assert manager.is_inference_only_phase(10100), "Phase 2 should be inference-only"
    assert not manager.is_inference_only_phase(15000), "Phase 3 is not inference-only"
    
    print("✅ All GradNorm control tests passed")
    return True

def test_parameter_counting():
    """Test parameter counting with adapter support."""
    print("\n🔢 Testing Parameter Counting...")
    
    model = DJMGNN(in_node_dim=10, hidden_dim=64, n_blocks=2, layers_per_block=2)
    manager = SimpleCurriculumManager(model)
    
    # Check that parameter counts are properly tracked
    assert 'pimeh' in manager.param_counts, "Should track PIMEH parameters"
    assert 'adapter' in manager.param_counts, "Should track adapter parameters"
    assert 'base' in manager.param_counts, "Should track base parameters"
    assert 'total' in manager.param_counts, "Should track total parameters"
    
    # Verify total equals sum of parts
    total_sum = manager.param_counts['pimeh'] + manager.param_counts['adapter'] + manager.param_counts['base']
    assert total_sum == manager.param_counts['total'], f"Parameter count mismatch: {total_sum} != {manager.param_counts['total']}"
    
    print("✅ Parameter counting test passed")
    return True

def test_phase_transitions():
    """Test phase transition logic and logging."""
    print("\n🔄 Testing Phase Transitions...")
    
    model = DJMGNN(in_node_dim=10, hidden_dim=64, n_blocks=2, layers_per_block=2)
    manager = SimpleCurriculumManager(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Test initial state
    assert manager.current_phase is None, "Should start with no current phase"
    
    # Test transition to Phase 0
    changed = manager.update_phase(0, optimizer)
    assert changed, "Should detect phase change on first update"
    assert manager.current_phase == 0, "Should be in Phase 0"
    
    # Test staying in Phase 0
    changed = manager.update_phase(4000, optimizer)
    assert not changed, "Should not detect phase change within Phase 0"
    assert manager.current_phase == 0, "Should still be in Phase 0"
    
    # Test transition to Phase 1
    changed = manager.update_phase(8000, optimizer)
    assert changed, "Should detect transition to Phase 1"
    assert manager.current_phase == 1, "Should be in Phase 1"
    
    # Test transition to Phase 2
    changed = manager.update_phase(10000, optimizer)
    assert changed, "Should detect transition to Phase 2"
    assert manager.current_phase == 2, "Should be in Phase 2"
    
    # Test transition to Phase 3
    changed = manager.update_phase(10200, optimizer)
    assert changed, "Should detect transition to Phase 3"
    assert manager.current_phase == 3, "Should be in Phase 3"
    
    print("✅ All phase transition tests passed")
    return True

def run_all_tests():
    """Run all Apollo curriculum tests."""
    print("🚀 Starting Project Apollo 4-Phase Curriculum Tests...")
    
    tests = [
        test_phase_constants,
        test_phase_logic,
        test_loss_weights,
        test_freezing_methods,
        test_gradnorm_control,
        test_parameter_counting,
        test_phase_transitions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Project Apollo curriculum is ready!")
        return True
    else:
        print("💥 Some tests failed! Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)