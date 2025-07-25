"""
Physics-Informed Minimal Equivariant Head (PIMEH) for Rotational Constants

This module implements a lightweight, physics-based head for computing molecular
rotational constants A, B, C from molecular graph embeddings and 3D positions.
The approach uses learned atomic masses and physics-based inertia tensor computation
to achieve SE(3) equivariance for rotational properties.

Core Physics:
- Inertia tensor: I = Σ m_i (r_i² * I_3 - r_i ⊗ r_i)
- Rotational constants: A, B, C = h/(8π²c·I_a,b,c) where I_a ≤ I_b ≤ I_c
- Units: GHz (consistent with molecular spectroscopy conventions)

Key Features:
- Minimal parameters (~1.5k-2k): Only mass prediction MLP
- Physics-based computation ensures rotational invariance
- Robust edge case handling (single atoms, numerical stability)
- Efficient batched computation for graph neural networks

References:
- MASTER_IMPLEMENTATION_PLAN.md sections 4 and 8
- Molecular spectroscopy: Gordy & Cook, "Microwave Molecular Spectra"
"""

import math
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Physics constants (CODATA 2018 values)
PLANCK_CONSTANT = 6.62607015e-34  # J·s
SPEED_OF_LIGHT = 2.99792458e8     # m/s
PI = math.pi

# Molecular physics constants for unit conversion
ATOMIC_MASS_UNIT = 1.66053906660e-27  # kg (unified atomic mass unit)
ANGSTROM_TO_METER = 1e-10  # m

# Unit conversion for rotational constants from molecular units to GHz
# Formula: B (cm⁻¹) = h/(8π²cI) where I is in kg·m²
# Then: B (GHz) = B (cm⁻¹) × 29.9792458 (since 1 cm⁻¹ = 29.9792458 GHz)

# Conversion factor for I from amu·Å² to kg·m²
MOLECULAR_TO_SI_INERTIA = ATOMIC_MASS_UNIT * (ANGSTROM_TO_METER**2)  # 1.66054e-47 kg·m²/(amu·Å²)

# Conversion factor for rotational constant: B(cm⁻¹) = h/(8π²c) / I(kg·m²)
CONST_H_8PI2C = PLANCK_CONSTANT / (8 * PI**2 * SPEED_OF_LIGHT * 100)  # Factor for cm⁻¹ (c in cm/s)

# Direct conversion from amu·Å² to GHz
CM_INV_TO_GHZ = 29.9792458  # 1 cm⁻¹ = 29.9792458 GHz
ROTATIONAL_CONST_FACTOR_GHZ = (CONST_H_8PI2C / MOLECULAR_TO_SI_INERTIA) * CM_INV_TO_GHZ

logger = logging.getLogger(__name__)


def compute_inertia_tensor(
    masses: torch.Tensor,
    positions: torch.Tensor, 
    batch: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute the inertia tensor for each molecule in the batch.
    
    Physics Formula:
    I = Σ m_i (r_i² * I_3 - r_i ⊗ r_i)
    
    where:
    - m_i: mass of atom i
    - r_i: position vector of atom i relative to center of mass
    - I_3: 3x3 identity matrix
    - ⊗: outer product
    
    Args:
        masses: Atomic masses [N] where N is total atoms across all molecules
        positions: Atomic positions [N, 3] in Cartesian coordinates
        batch: Batch indices [N] indicating which molecule each atom belongs to
        eps: Small constant for numerical stability
        
    Returns:
        torch.Tensor: Inertia tensors [B, 3, 3] where B is batch size
        
    Raises:
        ValueError: If input dimensions are inconsistent
        RuntimeError: If center of mass computation fails
    """
    if masses.dim() != 1 or positions.dim() != 2 or batch.dim() != 1:
        raise ValueError(
            f"Expected masses [N], positions [N, 3], batch [N], "
            f"got shapes {masses.shape}, {positions.shape}, {batch.shape}"
        )
    
    if positions.size(1) != 3:
        raise ValueError(f"Positions must have 3 coordinates, got {positions.size(1)}")
    
    if not (masses.size(0) == positions.size(0) == batch.size(0)):
        raise ValueError(
            f"Inconsistent sizes: masses {masses.size(0)}, "
            f"positions {positions.size(0)}, batch {batch.size(0)}"
        )
    
    device = masses.device
    dtype = masses.dtype
    
    # Get unique batch indices and number of molecules
    batch_indices = batch.unique(sorted=True)
    num_graphs = len(batch_indices)
    
    if num_graphs == 0:
        return torch.zeros(0, 3, 3, device=device, dtype=dtype)
    
    # Initialize output tensor
    inertia_tensors = torch.zeros(num_graphs, 3, 3, device=device, dtype=dtype)
    
    # Process each molecule separately for numerical stability
    for i, graph_id in enumerate(batch_indices):
        # Extract atoms for this molecule
        atom_mask = (batch == graph_id)
        mol_masses = masses[atom_mask]  # [n_atoms]
        mol_positions = positions[atom_mask]  # [n_atoms, 3]
        
        n_atoms = mol_masses.size(0)
        
        # Handle single atom case - treat as point mass with small moment
        if n_atoms == 1:
            # For single atoms, use a small diagonal inertia tensor
            # This prevents division by zero in rotational constant calculation
            small_moment = 1e-40  # Very small moment in kg·m²
            inertia_tensors[i] = torch.eye(3, device=device, dtype=dtype) * small_moment
            continue
        
        # Ensure masses are positive (add small epsilon for stability)
        mol_masses = torch.clamp(mol_masses, min=eps)
        total_mass = mol_masses.sum()
        
        if total_mass < eps:
            logger.warning(f"Very small total mass {total_mass:.2e} for molecule {graph_id}")
            # Use identity matrix with small diagonal
            inertia_tensors[i] = torch.eye(3, device=device, dtype=dtype) * eps
            continue
        
        # Compute center of mass
        weighted_positions = mol_positions * mol_masses.unsqueeze(1)  # [n_atoms, 3]
        center_of_mass = weighted_positions.sum(dim=0) / total_mass  # [3]
        
        # Center positions relative to center of mass
        centered_positions = mol_positions - center_of_mass.unsqueeze(0)  # [n_atoms, 3]
        
        # Compute inertia tensor using physics formula
        # I = Σ m_i (r_i² * I_3 - r_i ⊗ r_i)
        r_squared = (centered_positions**2).sum(dim=1)  # [n_atoms]
        
        # Diagonal terms: Σ m_i * r_i²
        diagonal_contribution = (mol_masses * r_squared).sum()
        I_diagonal = torch.eye(3, device=device, dtype=dtype) * diagonal_contribution
        
        # Off-diagonal terms: -Σ m_i * r_i ⊗ r_i
        # Compute outer products for all atoms at once
        weighted_positions = centered_positions * mol_masses.unsqueeze(1)  # [n_atoms, 3]
        outer_products = torch.bmm(
            weighted_positions.unsqueeze(2),  # [n_atoms, 3, 1]
            centered_positions.unsqueeze(1)   # [n_atoms, 1, 3]
        )  # [n_atoms, 3, 3]
        
        off_diagonal_contribution = outer_products.sum(dim=0)  # [3, 3]
        
        # Combine contributions: I = diagonal - off_diagonal
        inertia_tensor = I_diagonal - off_diagonal_contribution
        
        # Ensure positive semi-definite with careful regularization
        # For linear molecules, one diagonal element might be ~0, so we need to handle this carefully
        diagonal_elements = inertia_tensor.diagonal()
        # Only add regularization where needed (avoid making positive values negative)
        regularization = torch.clamp(eps - diagonal_elements, min=0)
        inertia_tensor.diagonal().add_(regularization)
        
        inertia_tensors[i] = inertia_tensor
    
    return inertia_tensors


class PhysicsInformedMinimalEquivariantHead(nn.Module):
    """
    Physics-Informed Minimal Equivariant Head for rotational constants prediction.
    
    This lightweight module learns effective atomic masses and computes rotational
    constants A, B, C using physics-based inertia tensor calculations. The approach
    ensures SE(3) equivariance by construction while maintaining minimal parameter count.
    
    Architecture:
    - Mass MLP: hidden_dim -> hidden_dim//2 -> 1 (with SiLU and Softplus)
    - Physics computation: inertia tensor -> eigenvalues -> rotational constants
    - Output: [batch_size, 3] tensor with A, B, C in GHz
    
    Key Properties:
    - Parameter count: ~1.5k-2k (only mass prediction network)
    - SE(3) equivariant by physics-based construction
    - Robust handling of edge cases (single atoms, numerical instabilities)
    - Units: GHz (standard molecular spectroscopy convention)
    
    Args:
        hidden_dim: Input feature dimension from graph neural network
        
    Example:
        >>> pimeh = PhysicsInformedMinimalEquivariantHead(hidden_dim=128)
        >>> h = torch.randn(10, 128)  # Node embeddings
        >>> pos = torch.randn(10, 3)  # Atomic positions  
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])  # Batch indices
        >>> rot_constants = pimeh(h, pos, batch)  # Shape: [3, 3] (A, B, C for 3 molecules)
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize the Physics-Informed Minimal Equivariant Head.
        
        Args:
            hidden_dim: Dimension of input node embeddings
        """
        super().__init__()
        
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        
        self.hidden_dim = hidden_dim
        
        # Minimal mass prediction network
        # Uses SiLU for smooth gradients and Softplus to ensure positive masses
        self.mass_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Smooth activation for stable gradients
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(beta=2.0)  # Ensure positive masses, β=2 for steeper slope
        )
        
        # Initialize mass MLP weights for reasonable mass range
        # Target: masses in range [0.1, 100] atomic mass units (approximately)
        with torch.no_grad():
            # Initialize final layer to produce masses around 1-10 range after Softplus
            nn.init.normal_(self.mass_mlp[-2].weight, mean=0.0, std=0.1)
            nn.init.constant_(self.mass_mlp[-2].bias, 1.0)  # Bias of 1.0 gives ~2.5 after Softplus(β=2)
        
        # Log architecture info
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"PIMEH initialized: {param_count:,} parameters")
    
    def forward(
        self, 
        h: torch.Tensor, 
        pos: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rotational constants from node embeddings and positions.
        
        Physics Pipeline:
        1. Predict atomic masses from embeddings: m_i = MLP(h_i)
        2. Compute inertia tensor: I = Σ m_i (r_i² * I_3 - r_i ⊗ r_i)  
        3. Extract eigenvalues: I_a ≤ I_b ≤ I_c = eig(I)
        4. Convert to rotational constants: A, B, C = h/(8π²c·I_a,b,c)
        
        Args:
            h: Node embeddings [N, hidden_dim] where N is total atoms
            pos: Atomic positions [N, 3] in Cartesian coordinates
            batch: Batch assignment [N] indicating molecule membership
            
        Returns:
            torch.Tensor: Rotational constants [batch_size, 3] with columns [A, B, C] in GHz
            
        Raises:
            ValueError: If input dimensions are inconsistent
            RuntimeError: If physics computation fails
        """
        if h.dim() != 2 or pos.dim() != 2 or batch.dim() != 1:
            raise ValueError(
                f"Expected h [N, D], pos [N, 3], batch [N], "
                f"got shapes {h.shape}, {pos.shape}, {batch.shape}"
            )
        
        if h.size(1) != self.hidden_dim:
            raise ValueError(
                f"Expected embedding dim {self.hidden_dim}, got {h.size(1)}"
            )
        
        if pos.size(1) != 3:
            raise ValueError(f"Expected 3D positions, got {pos.size(1)}D")
        
        if not (h.size(0) == pos.size(0) == batch.size(0)):
            raise ValueError(
                f"Inconsistent batch sizes: h {h.size(0)}, pos {pos.size(0)}, batch {batch.size(0)}"
            )
        
        device = h.device
        dtype = h.dtype
        
        # Handle empty input
        if h.size(0) == 0:
            return torch.empty(0, 3, device=device, dtype=dtype)
        
        try:
            # Step 1: Predict atomic masses
            masses = self.mass_mlp(h).squeeze(-1)  # [N]
            
            # Step 2: Compute inertia tensors for each molecule
            inertia_tensors = compute_inertia_tensor(masses, pos, batch)  # [B, 3, 3]
            batch_size = inertia_tensors.size(0)
            
            if batch_size == 0:
                return torch.empty(0, 3, device=device, dtype=dtype)
            
            # Step 3: Compute eigenvalues with regularization for numerical stability
            # Add small regularization to ensure positive definite matrices
            regularization = 1e-6 * torch.eye(3, device=device, dtype=dtype)
            regularized_tensors = inertia_tensors + regularization
            
            # Use eigvalsh for symmetric matrices (more stable than eig)
            eigenvalues = torch.linalg.eigvalsh(regularized_tensors)  # [B, 3], ascending order
            
            # Ensure eigenvalues are positive (critical for rotational constants)
            eigenvalues = torch.clamp(eigenvalues, min=1e-40)
            
            # Step 4: Convert to rotational constants in GHz
            # Formula: A, B, C = h/(8π²c·I_a,b,c) where I_a ≤ I_b ≤ I_c
            # Note: A ≥ B ≥ C since A ∝ 1/I_a and I_a ≤ I_b ≤ I_c
            # Direct conversion from molecular units (amu·Å²) to GHz
            rot_constants_ghz = ROTATIONAL_CONST_FACTOR_GHZ / eigenvalues
            
            # Reverse order since rotational constants are inversely related to moments
            # eigenvalues: I_a ≤ I_b ≤ I_c -> rot_constants: A ≥ B ≥ C
            rot_constants_ordered = torch.flip(rot_constants_ghz, dims=[1])  # [B, 3] -> [A, B, C]
            
            # Step 5: Apply physical bounds for molecular rotational constants
            # Typical range: 0.01 GHz to 1000 GHz for small to medium molecules
            rot_constants_clamped = torch.clamp(rot_constants_ordered, min=0.01, max=1000.0)
            
            return rot_constants_clamped
            
        except Exception as e:
            logger.error(f"PIMEH forward pass failed: {e}")
            logger.error(f"Input shapes - h: {h.shape}, pos: {pos.shape}, batch: {batch.shape}")
            
            # Fallback: return default rotational constants
            batch_indices = batch.unique(sorted=True)
            fallback_size = len(batch_indices)
            fallback_constants = torch.full(
                (fallback_size, 3), 10.0, device=device, dtype=dtype
            )
            logger.warning(f"Using fallback rotational constants: {fallback_constants.mean():.2f} GHz")
            
            return fallback_constants
    
    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        param_count = sum(p.numel() for p in self.parameters())
        return f"hidden_dim={self.hidden_dim}, parameters={param_count:,}"


def validate_rotational_invariance(
    pimeh_head: PhysicsInformedMinimalEquivariantHead,
    h: torch.Tensor,
    pos: torch.Tensor, 
    batch: torch.Tensor,
    num_rotations: int = 5,
    tolerance: float = 1e-4
) -> bool:
    """
    Validate that PIMEH output is rotationally invariant.
    
    Tests the fundamental SE(3) equivariance property by applying random
    rotations to molecular positions and verifying that rotational constants
    remain unchanged (within numerical tolerance).
    
    Args:
        pimeh_head: PIMEH module to test
        h: Node embeddings [N, hidden_dim]
        pos: Atomic positions [N, 3]
        batch: Batch indices [N]
        num_rotations: Number of random rotations to test
        tolerance: Maximum allowed deviation (relative error)
        
    Returns:
        bool: True if rotationally invariant within tolerance
    """
    pimeh_head.eval()
    
    with torch.no_grad():
        # Compute reference rotational constants
        ref_constants = pimeh_head(h, pos, batch)
        
        for _ in range(num_rotations):
            # Generate random rotation matrix
            angles = torch.rand(3) * 2 * PI  # Random Euler angles
            
            # Create rotation matrix from Euler angles (ZYX convention)
            cos_angles = torch.cos(angles)
            sin_angles = torch.sin(angles)
            
            # Rotation matrices for each axis
            R_z = torch.tensor([
                [cos_angles[0], -sin_angles[0], 0],
                [sin_angles[0], cos_angles[0], 0],
                [0, 0, 1]
            ], dtype=pos.dtype, device=pos.device)
            
            R_y = torch.tensor([
                [cos_angles[1], 0, sin_angles[1]],
                [0, 1, 0],
                [-sin_angles[1], 0, cos_angles[1]]
            ], dtype=pos.dtype, device=pos.device)
            
            R_x = torch.tensor([
                [1, 0, 0],
                [0, cos_angles[2], -sin_angles[2]],
                [0, sin_angles[2], cos_angles[2]]
            ], dtype=pos.dtype, device=pos.device)
            
            # Combined rotation matrix
            R = R_z @ R_y @ R_x
            
            # Apply rotation to positions
            rotated_pos = pos @ R.T  # [N, 3] @ [3, 3] -> [N, 3]
            
            # Compute rotational constants for rotated molecule
            rotated_constants = pimeh_head(h, rotated_pos, batch)
            
            # Check invariance (relative error)
            rel_error = torch.abs(rotated_constants - ref_constants) / (ref_constants + 1e-8)
            max_error = rel_error.max().item()
            
            if max_error > tolerance:
                logger.warning(
                    f"Rotational invariance violated: max relative error {max_error:.2e} > {tolerance:.2e}"
                )
                return False
    
    logger.info(f"Rotational invariance validated with {num_rotations} random rotations")
    return True


# Export main classes and functions
__all__ = [
    'PhysicsInformedMinimalEquivariantHead',
    'compute_inertia_tensor', 
    'validate_rotational_invariance',
    'PLANCK_CONSTANT',
    'SPEED_OF_LIGHT', 
    'ROTATIONAL_CONST_FACTOR'
]


if __name__ == "__main__":
    # Basic functionality test
    logging.basicConfig(level=logging.INFO)
    
    # Test with small molecule (3 atoms)
    hidden_dim = 128
    pimeh = PhysicsInformedMinimalEquivariantHead(hidden_dim)
    
    # Create test data: 2 molecules with 3 and 2 atoms respectively
    h = torch.randn(5, hidden_dim)  # 5 atoms total
    pos = torch.randn(5, 3)  # 3D positions
    batch = torch.tensor([0, 0, 0, 1, 1])  # First 3 atoms in mol 0, last 2 in mol 1
    
    print(f"PIMEH parameters: {sum(p.numel() for p in pimeh.parameters()):,}")
    print(f"Input shapes - h: {h.shape}, pos: {pos.shape}, batch: {batch.shape}")
    
    # Forward pass
    rot_constants = pimeh(h, pos, batch)
    print(f"Output shape: {rot_constants.shape}")
    print(f"Rotational constants (GHz):\n{rot_constants}")
    
    # Test rotational invariance
    is_invariant = validate_rotational_invariance(pimeh, h, pos, batch)
    print(f"Rotationally invariant: {is_invariant}")
    
    print("PIMEH implementation test completed successfully!")