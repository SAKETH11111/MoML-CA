"""
Test PIMEH physics computation with manually set masses.

This test verifies that the physics computation works correctly by
bypassing the MLP and setting realistic atomic masses directly.
"""

import torch
import math
from moml.models.mgnn.pimeh import compute_inertia_tensor, ROTATIONAL_CONST_FACTOR_GHZ

def test_physics_computation():
    """Test the physics computation with realistic atomic masses."""
    print("=" * 60)
    print("Testing PIMEH Physics Computation")
    print("=" * 60)
    
    # Create a simple diatomic molecule (like HCl)
    # H at (-0.5, 0, 0), Cl at (0.5, 0, 0) - bond length 1.0 Angstrom
    positions = torch.tensor([
        [-0.5, 0.0, 0.0],  # Hydrogen
        [0.5, 0.0, 0.0],   # Chlorine
    ], dtype=torch.float32)
    
    # Realistic atomic masses (in atomic mass units, roughly)
    # H ≈ 1 amu, Cl ≈ 35 amu
    masses = torch.tensor([1.0, 35.0], dtype=torch.float32)
    batch = torch.tensor([0, 0])  # Both atoms in molecule 0
    
    print(f"Test molecule: H-Cl")
    print(f"Positions:\n{positions}")
    print(f"Masses: {masses.tolist()}")
    print()
    
    # Compute inertia tensor
    inertia_tensor = compute_inertia_tensor(masses, positions, batch)
    print(f"Inertia tensor shape: {inertia_tensor.shape}")
    print(f"Inertia tensor:\n{inertia_tensor[0]}")
    
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(inertia_tensor[0])
    print(f"Eigenvalues (moments of inertia): {eigenvalues}")
    
    # Convert to rotational constants (direct conversion to GHz)
    rot_constants_ghz = ROTATIONAL_CONST_FACTOR_GHZ / eigenvalues
    
    print(f"Rotational constants (GHz): {rot_constants_ghz}")
    
    # For a diatomic molecule, two rotational constants should be identical
    # (perpendicular to bond axis) and one should be much larger (along bond axis)
    print()
    print("Analysis:")
    print(f"  B_perp = {rot_constants_ghz[1]:.2e} GHz")
    print(f"  B_parallel = {rot_constants_ghz[2]:.2e} GHz") 
    print(f"  Ratio B_parallel/B_perp = {rot_constants_ghz[2]/rot_constants_ghz[1]:.2e}")
    
    # Test with a non-linear molecule (water-like)
    print("\n" + "="*40)
    print("Testing non-linear molecule (H2O-like)")
    
    # Water geometry: O at origin, H atoms at realistic positions
    pos_water = torch.tensor([
        [0.0, 0.0, 0.0],      # Oxygen
        [0.96, 0.0, 0.0],     # Hydrogen 1
        [-0.24, 0.93, 0.0],   # Hydrogen 2 (104.5° angle)
    ], dtype=torch.float32)
    
    # Realistic masses: O ≈ 16 amu, H ≈ 1 amu
    masses_water = torch.tensor([16.0, 1.0, 1.0], dtype=torch.float32)
    batch_water = torch.tensor([0, 0, 0])
    
    inertia_water = compute_inertia_tensor(masses_water, pos_water, batch_water)
    eigenvals_water = torch.linalg.eigvalsh(inertia_water[0])
    rot_consts_water = ROTATIONAL_CONST_FACTOR_GHZ / eigenvals_water
    
    print(f"Water rotational constants (GHz): {rot_consts_water}")
    print(f"  A = {rot_consts_water[2]:.2e} GHz")
    print(f"  B = {rot_consts_water[1]:.2e} GHz") 
    print(f"  C = {rot_consts_water[0]:.2e} GHz")
    
    # Check if they follow expected ordering A > B > C
    is_ordered = rot_consts_water[2] > rot_consts_water[1] > rot_consts_water[0]
    print(f"  Proper ordering (A > B > C): {is_ordered}")
    
    print("\nPhysics computation test completed!")

if __name__ == "__main__":
    test_physics_computation()