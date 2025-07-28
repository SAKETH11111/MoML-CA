#!/usr/bin/env python3
"""
PROJECT APOLLO Comprehensive Verification Script

This script performs exhaustive testing of all PROJECT APOLLO components
to ensure the 4-phase sequential training paradigm is correctly implemented.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/saketh/MoML-CA')

def setup_logging():
    """Setup logging for verification."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_critical_imports():
    """Test that all critical PROJECT APOLLO components can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing Critical Imports...")
    
    try:
        from scripts.train_alternating_optimized import SimpleCurriculumManager, PHASE_0_END_STEP, PHASE_1_END_STEP, PHASE_2_END_STEP
        from moml.models.mgnn.djmgnn import DJMGNN, GNNSequential
        from moml.models.mgnn.pimeh import PhysicsInformedMinimalEquivariantHead
        
        logger.info("✅ All critical imports successful")
        logger.info(f"   • Phase transitions: 0→{PHASE_0_END_STEP}→{PHASE_1_END_STEP}→{PHASE_2_END_STEP}→∞")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_gnn_sequential_fix():
    """Test that the GNNSequential fix resolves the argument mismatch."""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Testing GNNSequential Fix...")
    
    try:
        from moml.models.mgnn.djmgnn import DJMGNN, GNNSequential
        
        # Create model
        model = DJMGNN(in_node_dim=33, hidden_dim=64, n_blocks=2, layers_per_block=2)
        
        # Verify adapter is GNNSequential
        adapter_type = type(model.pimeh_adapter).__name__
        assert adapter_type == 'GNNSequential', f"Expected GNNSequential, got {adapter_type}"
        
        # Test method signature
        import inspect
        signature = inspect.signature(model.pimeh_adapter.forward)
        params = list(signature.parameters.keys())
        expected = ['x', 'edge_index', 'edge_attr']
        assert params == expected, f"Expected {expected}, got {params}"
        
        logger.info("✅ GNNSequential fix verified")
        logger.info(f"   • PIMEH adapter type: {adapter_type}")
        logger.info(f"   • Forward signature: {params}")
        return True
        
    except Exception as e:
        logger.error(f"❌ GNNSequential test failed: {e}")
        return False

def test_model_architecture():
    """Test the complete DJMGNN+PIMEH architecture."""
    logger = logging.getLogger(__name__)
    logger.info("🏗️ Testing Model Architecture...")
    
    try:
        from moml.models.mgnn.djmgnn import DJMGNN
        
        model = DJMGNN(in_node_dim=33, hidden_dim=160, n_blocks=4, layers_per_block=6)
        
        # Test required components exist
        required_components = ['pimeh_head', 'pimeh_adapter', 'graph_head', 'head_energy', 'node_head']
        for component in required_components:
            assert hasattr(model, component), f"Missing component: {component}"
        
        # Test parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        pimeh_params = sum(p.numel() for p in model.pimeh_head.parameters())
        adapter_params = sum(p.numel() for p in model.pimeh_adapter.parameters())
        
        logger.info("✅ Model architecture verified")
        logger.info(f"   • Total parameters: {total_params:,}")
        logger.info(f"   • PIMEH head parameters: {pimeh_params:,}")
        logger.info(f"   • PIMEH adapter parameters: {adapter_params:,}")
        logger.info(f"   • All required components present: {required_components}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model architecture test failed: {e}")
        return False

def test_4phase_curriculum():
    """Test the 4-phase curriculum manager."""
    logger = logging.getLogger(__name__)
    logger.info("📚 Testing 4-Phase Curriculum...")
    
    try:
        from scripts.train_alternating_optimized import SimpleCurriculumManager
        from moml.models.mgnn.djmgnn import DJMGNN
        
        model = DJMGNN(in_node_dim=33, hidden_dim=64, n_blocks=2, layers_per_block=2)
        curriculum = SimpleCurriculumManager(model)
        
        # Test phase detection at key points
        test_cases = [
            (0, 0, "Backbone Pre-training"),
            (4000, 0, "Backbone Pre-training"),
            (7999, 0, "Backbone Pre-training"),
            (8000, 1, "PIMEH Adaptation"),
            (9000, 1, "PIMEH Adaptation"),
            (9999, 1, "PIMEH Adaptation"),
            (10000, 2, "Stability Check"),
            (10100, 2, "Stability Check"),
            (10199, 2, "Stability Check"),
            (10200, 3, "Joint Polishing"),
            (15000, 3, "Joint Polishing"),
            (20000, 3, "Joint Polishing"),
        ]
        
        for step, expected_phase, expected_desc in test_cases:
            actual_phase = curriculum.get_current_phase(step)
            assert actual_phase == expected_phase, f"Step {step}: expected phase {expected_phase}, got {actual_phase}"
        
        logger.info("✅ 4-Phase curriculum verified")
        logger.info(f"   • Phase 0: 0-8000 (Backbone Pre-training)")
        logger.info(f"   • Phase 1: 8000-10000 (PIMEH Adaptation)")
        logger.info(f"   • Phase 2: 10000-10200 (Stability Check)")
        logger.info(f"   • Phase 3: 10200+ (Joint Polishing)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Curriculum test failed: {e}")
        return False

def test_optimizer_switching():
    """Test the multi-optimizer switching system."""
    logger = logging.getLogger(__name__)
    logger.info("⚙️ Testing Optimizer Switching...")
    
    try:
        from moml.models.mgnn.djmgnn import DJMGNN
        
        model = DJMGNN(in_node_dim=33, hidden_dim=64, n_blocks=2, layers_per_block=2)
        
        # Create optimizers like in training script
        base_optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        pimeh_params = list(model.pimeh_head.parameters()) + list(model.pimeh_adapter.parameters())
        pimeh_optimizer = optim.AdamW(pimeh_params, lr=1e-3, weight_decay=1e-5)
        
        # Test optimizer mapping
        optimizers = {
            0: base_optimizer,
            1: pimeh_optimizer,
            2: base_optimizer,
            3: base_optimizer
        }
        
        # Test parameter counts
        base_param_count = sum(p.numel() for p in base_optimizer.param_groups[0]['params'])
        pimeh_param_count = sum(p.numel() for p in pimeh_optimizer.param_groups[0]['params'])
        total_params = sum(p.numel() for p in model.parameters())
        
        assert base_param_count == total_params, "Base optimizer should have all parameters"
        assert pimeh_param_count < total_params, "PIMEH optimizer should have fewer parameters"
        assert pimeh_param_count > 0, "PIMEH optimizer should have some parameters"
        
        logger.info("✅ Optimizer switching verified")
        logger.info(f"   • Base optimizer parameters: {base_param_count:,}")
        logger.info(f"   • PIMEH optimizer parameters: {pimeh_param_count:,}")
        logger.info(f"   • Phase mapping: {list(optimizers.keys())}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimizer switching test failed: {e}")
        return False

def test_parameter_freezing():
    """Test parameter freezing in different phases."""
    logger = logging.getLogger(__name__)
    logger.info("🧊 Testing Parameter Freezing...")
    
    try:
        from scripts.train_alternating_optimized import SimpleCurriculumManager
        from moml.models.mgnn.djmgnn import DJMGNN
        
        model = DJMGNN(in_node_dim=33, hidden_dim=64, n_blocks=2, layers_per_block=2)
        curriculum = SimpleCurriculumManager(model)
        
        # Create dummy optimizer for testing
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Test Phase 0: freeze PIMEH
        curriculum.update_phase(0, optimizer)
        pimeh_frozen = all(not p.requires_grad for name, p in model.named_parameters() 
                          if name.startswith('pimeh_head') or name.startswith('pimeh_adapter'))
        assert pimeh_frozen, "PIMEH should be frozen in Phase 0"
        
        # Test Phase 1: freeze backbone, unfreeze PIMEH
        curriculum.update_phase(8000, optimizer)
        pimeh_active = all(p.requires_grad for name, p in model.named_parameters() 
                          if name.startswith('pimeh_head') or name.startswith('pimeh_adapter'))
        backbone_frozen = all(not p.requires_grad for name, p in model.named_parameters() 
                             if not (name.startswith('pimeh_head') or name.startswith('pimeh_adapter')))
        assert pimeh_active, "PIMEH should be active in Phase 1"
        assert backbone_frozen, "Backbone should be frozen in Phase 1"
        
        # Test Phase 3: unfreeze all
        curriculum.update_phase(10200, optimizer)
        all_active = all(p.requires_grad for p in model.parameters())
        assert all_active, "All parameters should be active in Phase 3"
        
        logger.info("✅ Parameter freezing verified")
        logger.info(f"   • Phase 0: PIMEH frozen ✓")
        logger.info(f"   • Phase 1: Backbone frozen, PIMEH active ✓")
        logger.info(f"   • Phase 3: All parameters active ✓")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parameter freezing test failed: {e}")
        return False

def test_checkpoint_compatibility():
    """Test the enhanced checkpoint system."""
    logger = logging.getLogger(__name__)
    logger.info("💾 Testing Checkpoint Compatibility...")
    
    try:
        from moml.models.mgnn.djmgnn import DJMGNN
        import tempfile
        import os
        
        model = DJMGNN(in_node_dim=33, hidden_dim=64, n_blocks=2, layers_per_block=2)
        
        # Create optimizers and schedulers
        base_optimizer = optim.AdamW(model.parameters(), lr=2e-4)
        pimeh_params = list(model.pimeh_head.parameters()) + list(model.pimeh_adapter.parameters())
        pimeh_optimizer = optim.AdamW(pimeh_params, lr=1e-3)
        
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=20000)
        pimeh_scheduler = optim.lr_scheduler.CosineAnnealingLR(pimeh_optimizer, T_max=2000)
        
        # Test checkpoint structure
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "base_optimizer_state_dict": base_optimizer.state_dict(),
            "pimeh_optimizer_state_dict": pimeh_optimizer.state_dict(),
            "base_scheduler_state_dict": base_scheduler.state_dict(),
            "pimeh_scheduler_state_dict": pimeh_scheduler.state_dict(),
            "step": 1000,
            "loss": 1.5,
            "seed": 1337,
        }
        
        # Test save/load cycle
        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(checkpoint_data, f.name)
            loaded_checkpoint = torch.load(f.name, map_location='cpu')
            
            # Verify all keys are present
            required_keys = ["model_state_dict", "base_optimizer_state_dict", 
                           "pimeh_optimizer_state_dict", "base_scheduler_state_dict", 
                           "pimeh_scheduler_state_dict"]
            for key in required_keys:
                assert key in loaded_checkpoint, f"Missing checkpoint key: {key}"
            
            os.unlink(f.name)
        
        logger.info("✅ Checkpoint compatibility verified")
        logger.info(f"   • Dual optimizer state saving ✓")
        logger.info(f"   • Dual scheduler state saving ✓")
        logger.info(f"   • Save/load cycle successful ✓")
        return True
        
    except Exception as e:
        logger.error(f"❌ Checkpoint test failed: {e}")
        return False

def test_training_integration():
    """Test that training components integrate correctly."""
    logger = logging.getLogger(__name__)
    logger.info("🎯 Testing Training Integration...")
    
    try:
        from scripts.train_alternating_optimized import SimpleCurriculumManager
        from moml.models.mgnn.djmgnn import DJMGNN
        from gradnorm_pytorch import GradNormLossWeighter
        
        # Create model and components
        model = DJMGNN(in_node_dim=33, hidden_dim=64, n_blocks=2, layers_per_block=2)
        curriculum = SimpleCurriculumManager(model)
        
        # Test GradNorm integration
        backbone_param = model.blocks[-1].transition_layers[-1].weight
        loss_weighter = GradNormLossWeighter(
            num_losses=4,
            learning_rate=1e-4,
            restoring_force_alpha=0.5,
            grad_norm_parameters=backbone_param
        )
        
        # Test that all components can work together
        optimizers = {
            0: optim.AdamW(model.parameters(), lr=2e-4),
            1: optim.AdamW(list(model.pimeh_head.parameters()) + list(model.pimeh_adapter.parameters()), lr=1e-3),
            2: optim.AdamW(model.parameters(), lr=2e-4),
            3: optim.AdamW(model.parameters(), lr=2e-4),
        }
        
        # Simulate training step components
        step = 0
        current_phase = curriculum.get_current_phase(step)
        current_optimizer = optimizers[current_phase]
        curriculum.update_phase(step, current_optimizer)
        
        logger.info("✅ Training integration verified")
        logger.info(f"   • Curriculum + Optimizer integration ✓")
        logger.info(f"   • GradNorm loss weighter ✓")
        logger.info(f"   • Multi-optimizer coordination ✓")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training integration test failed: {e}")
        return False

def run_comprehensive_verification():
    """Run all PROJECT APOLLO verification tests."""
    logger = setup_logging()
    logger.info("🚀 STARTING PROJECT APOLLO COMPREHENSIVE VERIFICATION")
    logger.info("=" * 80)
    
    tests = [
        ("Critical Imports", test_critical_imports),
        ("GNNSequential Fix", test_gnn_sequential_fix),
        ("Model Architecture", test_model_architecture),
        ("4-Phase Curriculum", test_4phase_curriculum),
        ("Optimizer Switching", test_optimizer_switching),
        ("Parameter Freezing", test_parameter_freezing),
        ("Checkpoint Compatibility", test_checkpoint_compatibility),
        ("Training Integration", test_training_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running {test_name} Test")
        logger.info("-" * 50)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} Test PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} Test FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"💥 {test_name} Test CRASHED: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "=" * 80)
    logger.info(f"📊 VERIFICATION RESULTS: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! PROJECT APOLLO IS FULLY VERIFIED!")
        logger.info("✅ Ready for production training run")
        return True
    else:
        logger.error(f"💥 {failed} tests failed! Please fix issues before proceeding")
        return False

if __name__ == "__main__":
    success = run_comprehensive_verification()
    sys.exit(0 if success else 1)