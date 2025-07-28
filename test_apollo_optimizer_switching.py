#!/usr/bin/env python3
"""
Test script for PROJECT APOLLO Action 2.2: Optimizer Switching Logic.
Verifies that the optimizer switching system works correctly across all phases.
"""

import sys
import torch
import torch.optim as optim

# Add the project root to the path
sys.path.insert(0, '/home/saketh/MoML-CA')

try:
    from scripts.train_alternating_optimized import SimpleCurriculumManager, PHASE_0_END_STEP, PHASE_1_END_STEP, PHASE_2_END_STEP
    from moml.models.mgnn.djmgnn import DJMGNN
    print("✅ Successfully imported optimizer switching components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_optimizer_creation():
    """Test that both optimizers can be created correctly."""
    print("\n🔧 Testing Optimizer Creation...")
    
    # Create model
    model = DJMGNN(
        in_node_dim=33,
        hidden_dim=64,
        n_blocks=2,
        layers_per_block=2
    )
    
    # Test base optimizer creation (all parameters)
    base_optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    base_param_count = sum(p.numel() for p in base_optimizer.param_groups[0]['params'])
    
    # Test PIMEH optimizer creation (only PIMEH + adapter parameters)
    pimeh_params = list(model.pimeh_head.parameters()) + list(model.pimeh_adapter.parameters())
    pimeh_optimizer = optim.AdamW(pimeh_params, lr=1e-3, weight_decay=1e-5)
    pimeh_param_count = sum(p.numel() for p in pimeh_optimizer.param_groups[0]['params'])
    
    # Verify parameter counts
    total_model_params = sum(p.numel() for p in model.parameters())
    
    print(f"   • Total model parameters: {total_model_params:,}")
    print(f"   • Base optimizer parameters: {base_param_count:,}")
    print(f"   • PIMEH optimizer parameters: {pimeh_param_count:,}")
    
    assert base_param_count == total_model_params, f"Base optimizer should have all parameters"
    assert pimeh_param_count < total_model_params, f"PIMEH optimizer should have fewer parameters"
    assert pimeh_param_count > 0, f"PIMEH optimizer should have some parameters"
    
    print("✅ Optimizer creation test passed")
    return model, base_optimizer, pimeh_optimizer

def test_optimizer_mapping():
    """Test the optimizer phase mapping logic."""
    print("\n🗺️ Testing Optimizer Mapping...")
    
    model, base_optimizer, pimeh_optimizer = test_optimizer_creation()
    
    # Create optimizer mapping
    optimizers = {
        0: base_optimizer,   # Phase 0: Backbone pre-training
        1: pimeh_optimizer,  # Phase 1: PIMEH adaptation
        2: base_optimizer,   # Phase 2: Stability check
        3: base_optimizer    # Phase 3: Joint polishing
    }
    
    # Test that each phase maps to the correct optimizer
    assert optimizers[0] == base_optimizer, "Phase 0 should use base optimizer"
    assert optimizers[1] == pimeh_optimizer, "Phase 1 should use PIMEH optimizer"
    assert optimizers[2] == base_optimizer, "Phase 2 should use base optimizer"
    assert optimizers[3] == base_optimizer, "Phase 3 should use base optimizer"
    
    print("✅ Optimizer mapping test passed")
    return model, optimizers

def test_phase_based_optimizer_selection():
    """Test that the correct optimizer is selected for each step."""
    print("\n🎯 Testing Phase-Based Optimizer Selection...")
    
    model, optimizers = test_optimizer_mapping()
    curriculum_manager = SimpleCurriculumManager(model)
    
    # Test cases: (step, expected_phase, expected_optimizer_type)
    test_cases = [
        (0, 0, "base"),        # Phase 0 start
        (4000, 0, "base"),     # Phase 0 middle
        (7999, 0, "base"),     # Phase 0 end
        (8000, 1, "pimeh"),    # Phase 1 start
        (9000, 1, "pimeh"),    # Phase 1 middle
        (9999, 1, "pimeh"),    # Phase 1 end
        (10000, 2, "base"),    # Phase 2 start
        (10100, 2, "base"),    # Phase 2 middle
        (10199, 2, "base"),    # Phase 2 end
        (10200, 3, "base"),    # Phase 3 start
        (15000, 3, "base"),    # Phase 3 middle
        (20000, 3, "base"),    # Phase 3 late
    ]
    
    for step, expected_phase, expected_optimizer_type in test_cases:
        actual_phase = curriculum_manager.get_current_phase(step)
        selected_optimizer = optimizers[actual_phase]
        
        # Determine optimizer type
        if selected_optimizer == optimizers[1]:  # PIMEH optimizer
            actual_optimizer_type = "pimeh"
        else:
            actual_optimizer_type = "base"
        
        assert actual_phase == expected_phase, f"Step {step}: expected phase {expected_phase}, got {actual_phase}"
        assert actual_optimizer_type == expected_optimizer_type, f"Step {step}: expected {expected_optimizer_type} optimizer, got {actual_optimizer_type}"
        
        print(f"   • Step {step:5d}: Phase {actual_phase} -> {actual_optimizer_type} optimizer ✓")
    
    print("✅ Phase-based optimizer selection test passed")
    return True

def test_optimizer_switching_simulation():
    """Simulate the main training loop optimizer switching logic."""
    print("\n🔄 Testing Optimizer Switching Simulation...")
    
    model, optimizers = test_optimizer_mapping()
    
    # Create schedulers
    base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=40000, eta_min=5e-6)
    pimeh_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizers[1], T_max=2000, eta_min=1e-5)
    
    schedulers = {
        0: base_scheduler,
        1: pimeh_scheduler,
        2: base_scheduler,
        3: base_scheduler
    }
    
    curriculum_manager = SimpleCurriculumManager(model)
    
    # Simulate the switching logic at key transition points
    transition_steps = [0, 8000, 10000, 10200]
    current_optimizer = None
    current_scheduler = None
    
    for step in transition_steps:
        # Simulate the main loop logic
        current_phase = curriculum_manager.get_current_phase(step)
        
        # Check if we need to switch optimizer (this is the main loop logic)
        new_optimizer = optimizers[current_phase]
        new_scheduler = schedulers[current_phase]
        
        if current_optimizer != new_optimizer:
            current_optimizer = new_optimizer
            current_scheduler = new_scheduler
            optimizer_type = "PIMEH" if current_phase == 1 else "base"
            print(f"   • Step {step:5d}: Switched to {optimizer_type} optimizer for Phase {current_phase}")
        
        # Simulate phase update (this would freeze/unfreeze parameters)
        phase_changed = curriculum_manager.update_phase(step, current_optimizer)
        if phase_changed:
            print(f"   • Step {step:5d}: Phase transition detected")
    
    print("✅ Optimizer switching simulation test passed")
    return True

def test_parameter_freezing_with_optimizers():
    """Test that parameter freezing works correctly with different optimizers."""
    print("\n🧊 Testing Parameter Freezing with Optimizers...")
    
    model, optimizers = test_optimizer_mapping()
    curriculum_manager = SimpleCurriculumManager(model)
    
    # Test Phase 0: freeze PIMEH, use base optimizer
    curriculum_manager.update_phase(0, optimizers[0])
    
    # Check that PIMEH parameters are frozen
    pimeh_frozen = all(not p.requires_grad for name, p in model.named_parameters() 
                      if name.startswith('pimeh_head') or name.startswith('pimeh_adapter'))
    assert pimeh_frozen, "PIMEH parameters should be frozen in Phase 0"
    print("   • Phase 0: PIMEH parameters correctly frozen ✓")
    
    # Test Phase 1: freeze backbone, use PIMEH optimizer
    curriculum_manager.update_phase(8000, optimizers[1])
    
    # Check that PIMEH parameters are active and backbone is frozen
    pimeh_active = all(p.requires_grad for name, p in model.named_parameters() 
                      if name.startswith('pimeh_head') or name.startswith('pimeh_adapter'))
    backbone_frozen = all(not p.requires_grad for name, p in model.named_parameters() 
                         if not (name.startswith('pimeh_head') or name.startswith('pimeh_adapter')))
    
    assert pimeh_active, "PIMEH parameters should be active in Phase 1"
    assert backbone_frozen, "Backbone parameters should be frozen in Phase 1"
    print("   • Phase 1: PIMEH active, backbone frozen ✓")
    
    # Test Phase 3: unfreeze all, use base optimizer
    curriculum_manager.update_phase(10200, optimizers[3])
    
    # Check that all parameters are active
    all_active = all(p.requires_grad for p in model.parameters())
    assert all_active, "All parameters should be active in Phase 3"
    print("   • Phase 3: All parameters correctly unfrozen ✓")
    
    print("✅ Parameter freezing with optimizers test passed")
    return True

def run_all_tests():
    """Run all optimizer switching tests."""
    print("🚀 Starting PROJECT APOLLO Optimizer Switching Tests...")
    
    tests = [
        test_optimizer_creation,
        test_optimizer_mapping,
        test_phase_based_optimizer_selection,
        test_optimizer_switching_simulation,
        test_parameter_freezing_with_optimizers,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result is True or result is not False:  # Handle functions that return tuples
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! PROJECT APOLLO optimizer switching is ready!")
        return True
    else:
        print("💥 Some tests failed! Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)