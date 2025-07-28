class SimpleCurriculumManager:
    """
    Project Apollo 4-Phase Sequential Training Curriculum Manager.
    
    Phase 0 (0-8000 steps): Backbone pre-training (freeze PIMEH & adapter)
    Phase 1 (8000-10000 steps): PIMEH adaptation (freeze backbone, unfreeze PIMEH & adapter)
    Phase 2 (10000-10200 steps): Stability check (freeze everything)
    Phase 3 (10200+ steps): Joint polishing (unfreeze everything)
    """
    
    def __init__(self, model: DJMGNN):
        self.model = model
        self.current_phase = None  # Initialize to None so first update always triggers change
        
        # New 4-phase weight configuration for Project Apollo
        self.phase_weights = {
            0: {'physics': 0.0, 'others': 1.0},    # Phase 0: No physics, focus on backbone
            1: {'physics': 1.0, 'others': 0.0},    # Phase 1: Only physics, PIMEH adaptation
            2: {'physics': 0.0, 'others': 0.0},    # Phase 2: Inference only, stability check
            3: {'physics': 1.0, 'others': 1.0}     # Phase 3: Balanced joint polishing
        }
        
        # Track parameter counts for logging
        self._count_parameters()
        
    def _count_parameters(self):
        """Count parameters in different model components including the new adapter."""
        pimeh_count = sum(p.numel() for p in self.model.pimeh_head.parameters())
        
        # Count adapter parameters if it exists
        adapter_count = 0
        if hasattr(self.model, 'pimeh_adapter'):
            adapter_count = sum(p.numel() for p in self.model.pimeh_adapter.parameters())
        
        total_count = sum(p.numel() for p in self.model.parameters())
        base_count = total_count - pimeh_count - adapter_count
        
        self.param_counts = {
            'pimeh': pimeh_count,
            'adapter': adapter_count,
            'base': base_count,
            'total': total_count
        }
        
    def get_current_phase(self, step: int) -> int:
        """Determine current curriculum phase based on step (0-indexed phases for Project Apollo)."""
        if step < PHASE_0_END_STEP:
            return 0  # Backbone pre-training
        elif step < PHASE_1_END_STEP:
            return 1  # PIMEH adaptation
        elif step < PHASE_2_END_STEP:
            return 2  # Stability check
        else:
            return 3  # Joint polishing
    
    def freeze_pimeh_and_adapter(self):
        """Freeze PIMEH head and adapter parameters (Phase 0: Backbone pre-training)."""
        frozen_count = 0
        active_count = 0
        
        for name, param in self.model.named_parameters():
            if name.startswith('pimeh_head') or name.startswith('pimeh_adapter'):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                active_count += param.numel()
        
        logger.info(f"Phase 0: Frozen PIMEH+Adapter ({frozen_count:,}), active backbone ({active_count:,})")
        return frozen_count, active_count
    
    def freeze_backbone(self):
        """Freeze backbone, unfreeze PIMEH and adapter (Phase 1: PIMEH adaptation)."""
        frozen_count = 0
        active_count = 0
        
        for name, param in self.model.named_parameters():
            if name.startswith('pimeh_head') or name.startswith('pimeh_adapter'):
                param.requires_grad = True
                active_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        logger.info(f"Phase 1: Frozen backbone ({frozen_count:,}), active PIMEH+Adapter ({active_count:,})")
        return frozen_count, active_count
    
    def freeze_all(self):
        """Freeze all parameters (Phase 2: Stability check)."""
        frozen_count = 0
        
        for param in self.model.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        
        logger.info(f"Phase 2: All parameters frozen ({frozen_count:,}) - inference only")
        return frozen_count, 0
    
    def unfreeze_all(self):
        """Unfreeze all parameters (Phase 3: Joint polishing)."""
        active_count = 0
        
        for param in self.model.parameters():
            param.requires_grad = True
            active_count += param.numel()
        
        logger.info(f"Phase 3: All parameters unfrozen ({active_count:,}) - joint polishing")
        return 0, active_count
    
    def update_phase(self, step: int, optimizer: optim.Optimizer) -> bool:
        """
        Update training phase based on current step using Project Apollo 4-phase schedule.
        
        Returns:
            bool: True if phase changed, False otherwise
        """
        new_phase = self.get_current_phase(step)
        
        if new_phase != self.current_phase:
            old_phase = self.current_phase
            self.current_phase = new_phase
            
            # Apply parameter freezing based on new phase
            if new_phase == 0:
                frozen, active = self.freeze_pimeh_and_adapter()
                phase_desc = "Backbone Pre-training"
                reason = f"Step {step} < {PHASE_0_END_STEP} (Phase 0 threshold)"
            elif new_phase == 1:
                frozen, active = self.freeze_backbone()
                phase_desc = "PIMEH Adaptation"
                reason = f"Step {step} >= {PHASE_0_END_STEP} and < {PHASE_1_END_STEP} (Phase 1 threshold)"
            elif new_phase == 2:
                frozen, active = self.freeze_all()
                phase_desc = "Stability Check"
                reason = f"Step {step} >= {PHASE_1_END_STEP} and < {PHASE_2_END_STEP} (Phase 2 threshold)"
            else:  # Phase 3
                frozen, active = self.unfreeze_all()
                phase_desc = "Joint Polishing"
                reason = f"Step {step} >= {PHASE_2_END_STEP} (Phase 3 threshold)"
            
            # Update optimizer parameter groups (important for momentum/adam states)
            self._update_optimizer_groups(optimizer)
            
            # Log phase transition with Project Apollo context
            physics_weight = self.phase_weights[new_phase]['physics']
            others_weight = self.phase_weights[new_phase]['others']
            
            # Console notification
            logger.info("=" * 90)
            logger.info(f"🚀 PROJECT APOLLO PHASE TRANSITION: {old_phase} -> {new_phase}")
            logger.info(f"   Description: {phase_desc}")
            logger.info(f"   Frozen Parameters: {frozen:,}")
            logger.info(f"   Active Parameters: {active:,}")
            logger.info(f"   Physics Loss Weight: {physics_weight:.1f}")
            logger.info(f"   Other Loss Weight: {others_weight:.1f}")
            logger.info(f"   Reason: {reason}")
            logger.info("=" * 90)
            
            return True
        
        return False
    
    def _update_optimizer_groups(self, optimizer: optim.Optimizer):
        """Update optimizer parameter groups after freezing/unfreezing."""
        # Clear momentum/adam states for frozen parameters
        # This prevents stale gradients from affecting training
        active_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                active_params.append(param)
        
        # Update optimizer's param_groups
        if len(optimizer.param_groups) > 0:
            optimizer.param_groups[0]['params'] = active_params
        
        # Clear state for parameters that are no longer active
        # In PyTorch optimizers, state keys are the actual parameter tensors
        params_to_remove = []
        for param_tensor in optimizer.state.keys():
            if not param_tensor.requires_grad:
                params_to_remove.append(param_tensor)
        
        for param_tensor in params_to_remove:
            del optimizer.state[param_tensor]
    
    def get_loss_weights(self, step: int) -> Dict[str, float]:
        """Get phase-specific loss weights for Project Apollo schedule."""
        phase = self.get_current_phase(step)
        weights = self.phase_weights[phase]
        
        return {
            'physics_loss': weights['physics'],
            'node_loss': weights['others'],
            'graph_loss': weights['others'],
            'energy_loss': weights['others']
        }
    
    def should_skip_gradnorm(self, step: int) -> bool:
        """Determine if GradNorm should be skipped for this phase."""
        # Skip GradNorm in phases 0, 1, and 2, only use in phase 3 (joint polishing)
        return self.get_current_phase(step) < 3
    
    def is_inference_only_phase(self, step: int) -> bool:
        """Check if current phase is inference-only (Phase 2: Stability check)."""
        return self.get_current_phase(step) == 2