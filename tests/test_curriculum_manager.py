import torch
import torch.nn as nn
import pytest

import sys, types

# Stub torchvision.transforms.Compose to avoid heavy dependency
# Stub external heavy dependencies to avoid import errors during unit tests
for module_name, attrs in {
    'torchvision.transforms': {'Compose': lambda *args, **kwargs: list(args)},
    'gradnorm_pytorch': {'GradNormLossWeighter': object},
    'rich': {},
    'rich.console': {'Console': object},
    'rich.progress': {'Progress': object, 'BarColumn': object, 'TextColumn': object, 'TimeRemainingColumn': object, 'SpinnerColumn': object, 'MofNCompleteColumn': object},
    'rich.table': {'Table': object},
    'rich.live': {'Live': object},
    'rich.panel': {'Panel': object},
    'rich.columns': {'Columns': object},
    'rich.text': {'Text': object},
} .items():
    if module_name not in sys.modules:
        mod = types.ModuleType(module_name)
        for attr_name, attr_val in attrs.items():
            setattr(mod, attr_name, attr_val)
        sys.modules[module_name] = mod

from scripts.train_alternating_optimized import CurriculumManager, PHASE_1_END_STEP, PHASE_2_END_STEP


class DummyModel(nn.Module):
    """Minimal dummy DJMGNN-like model with distinct parameter groups for testing."""

    def __init__(self):
        super().__init__()
        # Pretend base network
        self.base_layer = nn.Linear(10, 10)
        # Pretend PIMEH head
        self.pimeh_head = nn.Linear(10, 3)

    def forward(self, x):  # pragma: no cover
        return self.pimeh_head(self.base_layer(x))


def _count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_manager():
    model = DummyModel()
    # Fake logger with required method
    class _Logger:
        def log_curriculum_transition(self, *args, **kwargs):
            pass
    return CurriculumManager(model, _Logger()), model


def test_phase_determination():
    manager, _ = create_manager()
    assert manager.get_current_phase(0) == 1
    assert manager.get_current_phase(PHASE_1_END_STEP - 1) == 1
    assert manager.get_current_phase(PHASE_1_END_STEP) == 2
    assert manager.get_current_phase(PHASE_2_END_STEP - 1) == 2
    assert manager.get_current_phase(PHASE_2_END_STEP) == 3
    assert manager.get_current_phase(999_999) == 3


def test_freezing_unfreezing():
    manager, model = create_manager()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # Phase 1 freeze base
    manager.update_phase(0, opt)  # to phase 1
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith('pimeh_head'))
    assert all(p.requires_grad for n, p in model.named_parameters() if n.startswith('pimeh_head'))
    active1 = _count_trainable(model)

    # Phase 2 freeze pimeh
    manager.update_phase(PHASE_1_END_STEP, opt)
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith('pimeh_head'))
    assert all(p.requires_grad for n, p in model.named_parameters() if not n.startswith('pimeh_head'))
    active2 = _count_trainable(model)
    assert active2 != active1

    # Phase 3 unfreeze all
    manager.update_phase(PHASE_2_END_STEP, opt)
    assert all(p.requires_grad for p in model.parameters())
    active3 = _count_trainable(model)
    assert active3 > active2


def test_loss_weights():
    manager, _ = create_manager()

    w1 = manager.get_loss_weights(0)
    w2 = manager.get_loss_weights(PHASE_1_END_STEP)
    w3 = manager.get_loss_weights(PHASE_2_END_STEP)

    assert w1['physics_loss'] > w1['node_loss']
    assert w2['physics_loss'] < w2['node_loss']
    assert w3['physics_loss'] == w3['node_loss']


def test_gradnorm_skip():
    manager, _ = create_manager()
    assert manager.should_skip_gradnorm(0)
    assert manager.should_skip_gradnorm(PHASE_1_END_STEP)
    assert not manager.should_skip_gradnorm(PHASE_2_END_STEP)
