"""
Realistic test of PIMEH with molecular-like geometries.

This script tests the PIMEH implementation with more realistic molecular
structures to verify that it produces reasonable rotational constants.
"""

import torch
import logging
from moml.models.mgnn.pimeh import PhysicsInformedMinimalEquivariantHead, validate_rotational_invariance

logging.basicConfig(level=logging.INFO)

def create_water_molecule():
    """Create a water molecule geometry (H2O)."""
    # Water molecule geometry (approximate)
    # O at origin, H atoms at realistic positions
    positions = torch.tensor([
        [0.0, 0.0, 0.0],      # Oxygen
        [0.96, 0.0, 0.0],     # Hydrogen 1  
        [-0.24, 0.93, 0.0],   # Hydrogen 2
    ], dtype=torch.float32)
    
    # Simple embeddings (would come from GNN in practice)
    embeddings = torch.randn(3, 128)
    
    # All atoms belong to molecule 0
    batch = torch.tensor([0, 0, 0])
    
    return embeddings, positions, batch

def create_methane_molecule():
    """Create a methane molecule geometry (CH4)."""
    # Tetrahedral methane geometry
    positions = torch.tensor([
        [0.0, 0.0, 0.0],        # Carbon at center
        [1.09, 1.09, 1.09],     # H1
        [1.09, -1.09, -1.09],   # H2  
        [-1.09, 1.09, -1.09],   # H3
        [-1.09, -1.09, 1.09],   # H4
    ], dtype=torch.float32)
    
    embeddings = torch.randn(5, 128)
    batch = torch.tensor([0, 0, 0, 0, 0])
    
    return embeddings, positions, batch

def create_linear_molecule():
    """Create a linear molecule (like CO2)."""
    # Linear arrangement: O-C-O
    positions = torch.tensor([
        [-1.16, 0.0, 0.0],  # O1
        [0.0, 0.0, 0.0],    # C (center)
        [1.16, 0.0, 0.0],   # O2
    ], dtype=torch.float32)
    
    embeddings = torch.randn(3, 128)
    batch = torch.tensor([0, 0, 0])
    
    return embeddings, positions, batch

def test_molecular_geometries():
    """Test PIMEH with different molecular geometries."""
    print("=" * 60)
    print("Testing PIMEH with Realistic Molecular Geometries")
    print("=" * 60)
    
    hidden_dim = 128
    pimeh = PhysicsInformedMinimalEquivariantHead(hidden_dim)
    
    print(f"PIMEH parameters: {sum(p.numel() for p in pimeh.parameters()):,}")
    print()
    
    # Test different molecules
    molecules = [
        ("Water (H2O)", create_water_molecule()),
        ("Methane (CH4)", create_methane_molecule()), 
        ("Linear (CO2-like)", create_linear_molecule())
    ]
    
    for name, (h, pos, batch) in molecules:
        print(f"Testing {name}:")
        try:
            # Compute rotational constants
            rot_constants = pimeh(h, pos, batch)
            
            print(f"  Input shape: {h.shape[0]} atoms")
            print(f"  Rotational constants (GHz): A={rot_constants[0,0]:.4f}, B={rot_constants[0,1]:.4f}, C={rot_constants[0,2]:.4f}")
            
            # Check if constants follow expected ordering (A >= B >= C)
            constants = rot_constants[0].tolist()
            is_ordered = constants[0] >= constants[1] >= constants[2]
            print(f"  Properly ordered (A>=B>=C): {is_ordered}")
            
            # Test rotational invariance
            is_invariant = validate_rotational_invariance(pimeh, h, pos, batch, num_rotations=3)
            print(f"  SE(3) equivariant: {is_invariant}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()
    
    # Test batch processing with multiple molecules
    print("Testing batch processing:")
    try:
        h1, pos1, _ = create_water_molecule()
        h2, pos2, _ = create_methane_molecule()
        
        # Combine into batch
        h_batch = torch.cat([h1, h2], dim=0)  # 8 atoms total
        pos_batch = torch.cat([pos1, pos2], dim=0)
        batch_batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])  # 3 + 5 atoms
        
        rot_constants_batch = pimeh(h_batch, pos_batch, batch_batch)
        
        print(f"  Batch input: {h_batch.shape[0]} atoms, 2 molecules")
        print(f"  Batch output shape: {rot_constants_batch.shape}")
        print(f"  Molecule 1 constants: A={rot_constants_batch[0,0]:.4f}, B={rot_constants_batch[0,1]:.4f}, C={rot_constants_batch[0,2]:.4f}")
        print(f"  Molecule 2 constants: A={rot_constants_batch[1,0]:.4f}, B={rot_constants_batch[1,1]:.4f}, C={rot_constants_batch[1,2]:.4f}")
        
    except Exception as e:
        print(f"  Batch ERROR: {e}")
    
    print("\nRealistic molecular test completed!")

if __name__ == "__main__":
    test_molecular_geometries()