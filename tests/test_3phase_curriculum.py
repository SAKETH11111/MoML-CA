"""
Test 3-Phase Curriculum Training Implementation

This script validates the comprehensive 3-phase curriculum training system:

Phase 1 (0-2000 steps): Train only PIMEH, freeze base DJMGNN
Phase 2 (2000-6000 steps): Train base DJMGNN, freeze PIMEH  
Phase 3 (6000+ steps): Joint training of everything

Key Validation Areas:
1. Parameter freezing/unfreezing functions
2. Phase transition logic with step-based triggers
3. Phase-specific loss weighting
4. Curriculum tracking and logging
5. Integration with training loop
6. Optimizer state management during transitions

Usage:
    python test_3phase_curriculum.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os
from pathlib import Path
from typing import Dict
import tempfile
import time

# Add project root to path
sys.path.insert(0, '/home/saketh/MoML-CA')

from moml.models.mgnn.djmgnn import DJMGNN
from gradnorm_pytorch import GradNormLossWeighter

# Import from training script
sys.path.insert(0, '/home/saketh/MoML-CA/scripts')
from train_alternating_optimized import (
    CurriculumManager, EnhancedTrainingLogger, 
    PHASE_1_END_STEP, PHASE_2_END_STEP
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_model(device: str = "cpu") -> DJMGNN:
    """Create a test DJMGNN model for curriculum testing."""
    model = DJMGNN(
        in_node_dim=29,
        hidden_dim=64,  # Smaller for testing
        n_blocks=2,
        layers_per_block=2,
        graph_output_dims=19,
        node_output_dims=3,
        energy_output_dims=1
    ).to(device)
    return model


def create_test_batch(batch_size: int = 2, num_nodes: int = 8, device: str = "cpu"):
    """Create a mock batch for testing."""
    class MockBatch:
        def __init__(self):
            self.x = torch.randn(num_nodes, 29, device=device)
            self.edge_index = torch.tensor([
                [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]
            ], dtype=torch.long, device=device)
            self.batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=device)
            self.dist = torch.rand(self.edge_index.size(1), 1, device=device) * 3.0 + 1.0
            self.pos = torch.randn(num_nodes, 3, device=device)  # Crucial for PIMEH
            self.node_y = torch.randn(num_nodes, 3, device=device)
            self.y = torch.randn(batch_size, 19, device=device)
            self.y_graph = torch.randn(batch_size, 1, device=device)
        
        def to(self, device):
            return self  # Mock implementation
            
    return MockBatch()


def test_parameter_counting():
    """Test parameter counting in model components."""
    logger.info("Testing parameter counting functionality...")
    
    device = "cpu"
    model = create_test_model(device)
    
    # Create mock enhanced logger
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=1000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Validate parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        pimeh_params = sum(p.numel() for p in model.pimeh_head.parameters())
        base_params = total_params - pimeh_params
        
        assert curriculum_manager.param_counts['total'] == total_params
        assert curriculum_manager.param_counts['pimeh'] == pimeh_params
        assert curriculum_manager.param_counts['base'] == base_params
        
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   PIMEH parameters: {pimeh_params:,}")
        logger.info(f"   Base parameters: {base_params:,}")
        logger.info("✅ Parameter counting test passed!")


def test_phase_determination():
    """Test phase determination based on step numbers."""
    logger.info("Testing phase determination logic...")
    
    device = "cpu"
    model = create_test_model(device)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=10000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Test phase boundaries
        test_cases = [
            (0, 1), (1000, 1), (1999, 1),      # Phase 1
            (2000, 2), (3000, 2), (5999, 2),  # Phase 2
            (6000, 3), (8000, 3), (10000, 3)  # Phase 3
        ]
        
        for step, expected_phase in test_cases:
            actual_phase = curriculum_manager.get_current_phase(step)
            assert actual_phase == expected_phase, f"Step {step}: expected phase {expected_phase}, got {actual_phase}"
        
        logger.info("✅ Phase determination test passed!")


def test_parameter_freezing():
    """Test parameter freezing/unfreezing functionality."""
    logger.info("Testing parameter freezing functionality...")
    
    device = "cpu"
    model = create_test_model(device)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=10000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Test Phase 1: freeze base, unfreeze PIMEH
        frozen, active = curriculum_manager.freeze_base_model()
        
        pimeh_trainable = all(p.requires_grad for p in model.pimeh_head.parameters())
        base_frozen = all(not p.requires_grad for name, p in model.named_parameters() 
                         if not name.startswith('pimeh_head'))
        
        assert pimeh_trainable, "PIMEH parameters should be trainable in phase 1"
        assert base_frozen, "Base model parameters should be frozen in phase 1"
        logger.info(f"   Phase 1: {frozen:,} frozen, {active:,} active")
        
        # Test Phase 2: freeze PIMEH, unfreeze base
        frozen, active = curriculum_manager.freeze_pimeh()
        
        pimeh_frozen = all(not p.requires_grad for p in model.pimeh_head.parameters())
        base_trainable = all(p.requires_grad for name, p in model.named_parameters() 
                           if not name.startswith('pimeh_head'))
        
        assert pimeh_frozen, "PIMEH parameters should be frozen in phase 2"
        assert base_trainable, "Base model parameters should be trainable in phase 2"
        logger.info(f"   Phase 2: {frozen:,} frozen, {active:,} active")
        
        # Test Phase 3: unfreeze all
        frozen, active = curriculum_manager.unfreeze_all()
        
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable, "All parameters should be trainable in phase 3"
        logger.info(f"   Phase 3: {frozen:,} frozen, {active:,} active")
        
        logger.info("✅ Parameter freezing test passed!")


def test_phase_transitions():
    """Test automatic phase transitions during training."""
    logger.info("Testing phase transition logic...")
    
    device = "cpu"
    model = create_test_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=10000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Test phase transitions at boundaries
        transition_steps = [0, PHASE_1_END_STEP, PHASE_2_END_STEP, 8000]
        expected_phases = [1, 2, 3, 3]
        
        for step, expected_phase in zip(transition_steps, expected_phases):
            phase_changed = curriculum_manager.update_phase(step, optimizer)
            
            if step == 0:
                assert phase_changed, f"Initial phase setup should trigger change at step {step}"
            elif step in [PHASE_1_END_STEP, PHASE_2_END_STEP]:
                assert phase_changed, f"Phase transition should occur at step {step}"
            else:
                assert not phase_changed, f"No phase change should occur at step {step}"
            
            assert curriculum_manager.current_phase == expected_phase, \
                f"Step {step}: expected phase {expected_phase}, got {curriculum_manager.current_phase}"
        
        logger.info("✅ Phase transition test passed!")


def test_loss_weighting():
    """Test phase-specific loss weighting."""
    logger.info("Testing phase-specific loss weighting...")
    
    device = "cpu"
    model = create_test_model(device)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=10000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Test weights for each phase
        test_cases = [
            (1000, 1, {'physics': 2.0, 'others': 0.1}),  # Phase 1
            (3000, 2, {'physics': 0.1, 'others': 1.0}),  # Phase 2
            (7000, 3, {'physics': 1.0, 'others': 1.0})   # Phase 3
        ]
        
        for step, expected_phase, expected_weights in test_cases:
            weights = curriculum_manager.get_loss_weights(step)
            phase = curriculum_manager.get_current_phase(step)
            
            assert phase == expected_phase, f"Step {step}: expected phase {expected_phase}, got {phase}"
            assert weights['physics_loss'] == expected_weights['physics']
            assert weights['node_loss'] == expected_weights['others']
            assert weights['graph_loss'] == expected_weights['others']
            assert weights['energy_loss'] == expected_weights['others']
            
            logger.info(f"   Step {step} (Phase {phase}): physics={weights['physics_loss']:.1f}, others={weights['node_loss']:.1f}")
        
        logger.info("✅ Loss weighting test passed!")


def test_gradnorm_skipping():
    """Test GradNorm skipping logic."""
    logger.info("Testing GradNorm skipping logic...")
    
    device = "cpu"
    model = create_test_model(device)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=10000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Test GradNorm skipping for each phase
        test_cases = [
            (1000, True),   # Phase 1: skip GradNorm
            (3000, True),   # Phase 2: skip GradNorm  
            (7000, False)   # Phase 3: use GradNorm
        ]
        
        for step, should_skip in test_cases:
            skip_gradnorm = curriculum_manager.should_skip_gradnorm(step)
            phase = curriculum_manager.get_current_phase(step)
            
            assert skip_gradnorm == should_skip, \
                f"Step {step} (Phase {phase}): expected skip={should_skip}, got {skip_gradnorm}"
            
            logger.info(f"   Step {step} (Phase {phase}): skip_gradnorm={skip_gradnorm}")
        
        logger.info("✅ GradNorm skipping test passed!")


def test_optimizer_state_management():
    """Test optimizer state management during phase transitions."""
    logger.info("Testing optimizer state management...")
    
    device = "cpu"
    model = create_test_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Build up some optimizer state by doing a few steps
    batch = create_test_batch(device=device)
    
    for _ in range(5):
        optimizer.zero_grad()
        output = model(
            x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
            dist=batch.dist, pos=batch.pos
        )
        loss = nn.MSELoss()(output["graph_pred"], batch.y)
        loss.backward()
        optimizer.step()
    
    initial_state_keys = set(optimizer.state.keys())
    assert len(initial_state_keys) > 0, "Optimizer should have some state after training steps"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=10000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Test phase transition clears appropriate optimizer states
        curriculum_manager.update_phase(PHASE_1_END_STEP, optimizer)  # Transition to phase 2
        
        # Check that optimizer param_groups are updated
        active_param_ids = {id(p) for p in optimizer.param_groups[0]['params']}
        expected_active_ids = {id(p) for p in model.parameters() if p.requires_grad}
        
        assert active_param_ids == expected_active_ids, \
            "Optimizer param_groups should match active parameters"
        
        logger.info(f"   Active parameters after phase transition: {len(active_param_ids)}")
        logger.info("✅ Optimizer state management test passed!")


def test_curriculum_logging():
    """Test curriculum logging functionality."""
    logger.info("Testing curriculum logging...")
    
    device = "cpu"
    model = create_test_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=10000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Trigger phase transitions and check logging
        transition_steps = [0, PHASE_1_END_STEP, PHASE_2_END_STEP]
        
        for step in transition_steps:
            curriculum_manager.update_phase(step, optimizer)
        
        # Check that curriculum CSV was created and has data
        curriculum_csv_path = enhanced_logger.curriculum_csv_path
        assert curriculum_csv_path.exists(), "Curriculum CSV file should be created"
        
        with open(curriculum_csv_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 1, "Curriculum CSV should have header + data lines"
            assert 'curriculum_phase' in lines[0], "Curriculum CSV should have curriculum_phase column"
        
        logger.info(f"   Curriculum CSV created: {curriculum_csv_path.name}")
        logger.info(f"   Phase transitions logged: {len(enhanced_logger.phase_transitions)}")
        logger.info("✅ Curriculum logging test passed!")


def test_integration_with_training():
    """Test integration with actual training step."""
    logger.info("Testing integration with training step...")
    
    device = "cpu"
    model = create_test_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Get backbone parameter for GradNorm
    backbone_parameter = model.blocks[-1].transition_layers[-1].weight
    loss_weighter = GradNormLossWeighter(
        num_losses=4,
        learning_rate=1e-4,
        restoring_force_alpha=0.5,
        grad_norm_parameters=backbone_parameter
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_logger = EnhancedTrainingLogger(max_steps=10000, log_dir=temp_dir)
        curriculum_manager = CurriculumManager(model, enhanced_logger)
        
        # Import train_step function
        from train_alternating_optimized import train_step
        
        batch = create_test_batch(device=device)
        
        # Test training steps in different phases
        test_steps = [1000, 3000, 7000]  # One from each phase
        
        for step in test_steps:
            curriculum_manager.update_phase(step, optimizer)
            
            losses = train_step(
                model=model,
                optimizer=optimizer,
                loss_weighter=loss_weighter,
                batch=batch,
                device=torch.device(device),
                task_type="graph",
                curriculum_manager=curriculum_manager,
                step=step
            )
            
            # Validate loss structure
            assert 'total_loss' in losses
            assert 'physics_loss' in losses
            assert 'weights' in losses
            assert len(losses['weights']) == 4  # 4 loss components
            
            assert torch.isfinite(torch.tensor(losses['total_loss'])), \
                f"Total loss should be finite at step {step}"
            
            phase = curriculum_manager.get_current_phase(step)
            logger.info(f"   Step {step} (Phase {phase}): total_loss={losses['total_loss']:.4f}")
        
        logger.info("✅ Training integration test passed!")


def run_comprehensive_curriculum_tests():
    """Run all curriculum training tests."""
    logger.info("Starting comprehensive 3-phase curriculum tests...")
    logger.info("=" * 80)
    
    tests = [
        test_parameter_counting,
        test_phase_determination,
        test_parameter_freezing,
        test_phase_transitions,
        test_loss_weighting,
        test_gradnorm_skipping,
        test_optimizer_state_management,
        test_curriculum_logging,
        test_integration_with_training
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
    
    logger.info("=" * 80)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 ALL CURRICULUM TESTS PASSED!")
        logger.info("\nCurriculum Training System Ready:")
        logger.info(f"   - Phase 1 (0-{PHASE_1_END_STEP}): PIMEH Only Training")
        logger.info(f"   - Phase 2 ({PHASE_1_END_STEP}-{PHASE_2_END_STEP}): Base DJMGNN Training")
        logger.info(f"   - Phase 3 ({PHASE_2_END_STEP}+): Joint Training")
        logger.info("   - Parameter freezing/unfreezing: ✅")
        logger.info("   - Phase-specific loss weighting: ✅")
        logger.info("   - Automatic phase transitions: ✅")
        logger.info("   - Curriculum logging: ✅")
        logger.info("   - Training integration: ✅")
        logger.info("\n🚀 Ready for 40K step training run!")
    else:
        logger.error(f"💥 {failed} tests failed. Please fix issues before training.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_comprehensive_curriculum_tests()
    exit(0 if success else 1)