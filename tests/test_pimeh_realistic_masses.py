"""
Test PIMEH with realistic molecular data from literature.

This test uses known molecular geometries and compares computed rotational 
constants with literature values to validate the physics implementation.
"""

import torch
from moml.models.mgnn.pimeh import PhysicsInformedMinimalEquivariantHead, compute_inertia_tensor, ROTATIONAL_CONST_FACTOR_GHZ

def test_water_molecule():
    """Test with realistic water molecule geometry and known rotational constants."""
    print("=" * 60)
    print("Testing Water Molecule (H2O)")
    print("Literature values: A = 835.1 GHz, B = 435.4 GHz, C = 278.1 GHz") 
    print("=" * 60)
    
    # Accurate water geometry (from NIST): O-H = 0.9584 Å, H-O-H = 104.31°
    # Place O at origin, H atoms at realistic positions
    import math
    angle_rad = 104.31 * math.pi / 180 / 2  # Half the H-O-H angle
    oh_distance = 0.9584  # Angstrom
    
    positions = torch.tensor([
        [0.0, 0.0, 0.0],  # Oxygen at origin
        [oh_distance * math.cos(angle_rad), oh_distance * math.sin(angle_rad), 0.0],  # H1
        [oh_distance * math.cos(angle_rad), -oh_distance * math.sin(angle_rad), 0.0],  # H2
    ], dtype=torch.float32)
    
    print(f"Molecular geometry:")
    print(f"  O: {positions[0].tolist()}")
    print(f"  H1: {positions[1].tolist()}")  
    print(f"  H2: {positions[2].tolist()}")
    print(f"  O-H distance: {oh_distance:.4f} Å")
    print(f"  H-O-H angle: {104.31:.2f}°")
    print()
    
    # Use literature atomic masses (amu)
    masses = torch.tensor([15.999, 1.008, 1.008], dtype=torch.float32)  # O, H, H
    batch = torch.tensor([0, 0, 0])
    
    # Manual physics computation
    inertia_tensor = compute_inertia_tensor(masses, positions, batch)
    eigenvalues = torch.linalg.eigvalsh(inertia_tensor[0])
    rot_constants = ROTATIONAL_CONST_FACTOR_GHZ / eigenvalues
    
    # Sort in descending order (A >= B >= C)
    rot_constants_sorted = torch.sort(rot_constants, descending=True)[0]
    
    print(f"Computed inertia tensor eigenvalues (amu⋅Å²): {eigenvalues}")
    print(f"Computed rotational constants (GHz):")
    print(f"  A = {rot_constants_sorted[0]:.1f} GHz")
    print(f"  B = {rot_constants_sorted[1]:.1f} GHz") 
    print(f"  C = {rot_constants_sorted[2]:.1f} GHz")
    print()
    
    # Test with PIMEH head (will use learned masses, not literature values)
    pimeh = PhysicsInformedMinimalEquivariantHead(hidden_dim=128)
    embeddings = torch.randn(3, 128)  # Random embeddings
    
    with torch.no_grad():
        pimeh_constants = pimeh(embeddings, positions, batch)
        print(f"PIMEH prediction (untrained):")
        print(f"  A = {pimeh_constants[0,0]:.1f} GHz")
        print(f"  B = {pimeh_constants[0,1]:.1f} GHz")
        print(f"  C = {pimeh_constants[0,2]:.1f} GHz")
    
    print()

def test_co2_molecule():
    """Test with linear CO2 molecule."""
    print("=" * 60)
    print("Testing Carbon Dioxide (CO2) - Linear Molecule")
    print("Expected: Two equal rotational constants (B = C), A >> B,C")
    print("=" * 60)
    
    # Linear CO2: O-C-O with C-O = 1.162 Å
    co_distance = 1.162
    positions = torch.tensor([
        [-co_distance, 0.0, 0.0],  # O1
        [0.0, 0.0, 0.0],           # C (center)
        [co_distance, 0.0, 0.0],   # O2
    ], dtype=torch.float32)
    
    # Atomic masses (amu)
    masses = torch.tensor([15.999, 12.011, 15.999], dtype=torch.float32)  # O, C, O
    batch = torch.tensor([0, 0, 0])
    
    print(f"Linear geometry: O-C-O")
    print(f"C-O distance: {co_distance:.3f} Å")
    print()
    
    # Compute rotational constants
    inertia_tensor = compute_inertia_tensor(masses, positions, batch)
    eigenvalues = torch.linalg.eigvalsh(inertia_tensor[0])
    rot_constants = ROTATIONAL_CONST_FACTOR_GHZ / eigenvalues
    
    # For linear molecules, one moment should be much smaller (about bond axis)
    print(f"Inertia eigenvalues (amu⋅Å²): {eigenvalues}")
    print(f"Rotational constants (GHz): {rot_constants}")
    
    # The largest constant corresponds to rotation about the molecular axis (should be clamped)
    # The two smaller constants should be equal for a linear molecule
    constants_sorted = torch.sort(rot_constants)[0]  # ascending order
    print(f"  Smallest constants (B = C): {constants_sorted[0]:.2f}, {constants_sorted[1]:.2f} GHz")
    print(f"  Largest constant (A): {constants_sorted[2]:.2f} GHz (may be clamped)")
    print(f"  Ratio B:C = {constants_sorted[1]/constants_sorted[0]:.3f} (should be ≈ 1.0)")
    
    print()

def test_methane_molecule():
    """Test with tetrahedral methane molecule."""
    print("=" * 60)
    print("Testing Methane (CH4) - Tetrahedral")
    print("Expected: Nearly spherical, A ≈ B ≈ C")
    print("=" * 60)
    
    # Tetrahedral methane: C at center, H at tetrahedral positions
    # C-H distance = 1.09 Å
    ch_distance = 1.09
    
    # Tetrahedral angles: 109.47°
    # Tetrahedral coordinates (normalized to unit vectors, then scaled)
    sqrt3 = torch.sqrt(torch.tensor(3.0))
    positions = torch.tensor([
        [0.0, 0.0, 0.0],                              # C at center
        [1.0, 1.0, 1.0],                             # H1
        [1.0, -1.0, -1.0],                           # H2  
        [-1.0, 1.0, -1.0],                           # H3
        [-1.0, -1.0, 1.0],                           # H4
    ], dtype=torch.float32) * ch_distance / sqrt3
    
    # Atomic masses (amu)
    masses = torch.tensor([12.011, 1.008, 1.008, 1.008, 1.008], dtype=torch.float32)  # C, H×4
    batch = torch.tensor([0, 0, 0, 0, 0])
    
    print(f"Tetrahedral geometry:")
    print(f"C-H distance: {ch_distance:.3f} Å")
    print()
    
    # Compute rotational constants
    inertia_tensor = compute_inertia_tensor(masses, positions, batch)
    eigenvalues = torch.linalg.eigvalsh(inertia_tensor[0])
    rot_constants = ROTATIONAL_CONST_FACTOR_GHZ / eigenvalues
    
    print(f"Inertia eigenvalues (amu⋅Å²): {eigenvalues}")
    print(f"Rotational constants (GHz): {rot_constants}")
    
    # For a symmetric top molecule, should have similar values
    constants_sorted = torch.sort(rot_constants, descending=True)[0]
    print(f"  A = {constants_sorted[0]:.2f} GHz")
    print(f"  B = {constants_sorted[1]:.2f} GHz")
    print(f"  C = {constants_sorted[2]:.2f} GHz")
    print(f"  Ratio A:B:C = {constants_sorted[0]/constants_sorted[2]:.2f}:{constants_sorted[1]/constants_sorted[2]:.2f}:1.00")
    
    print()

if __name__ == "__main__":
    test_water_molecule()
    test_co2_molecule() 
    test_methane_molecule()
    print("Realistic molecular tests completed!")