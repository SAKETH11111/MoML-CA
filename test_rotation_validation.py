"""
Simple validation script for rotation invariance test suite.
Tests core functionality without heavy dependencies.
"""

import torch
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

# Direct imports to avoid dependency issues
from moml.models.mgnn.pimeh import PhysicsInformedMinimalEquivariantHead, compute_inertia_tensor

def test_basic_rotation_invariance():
    """Test basic rotation invariance with water molecule."""
    print("Testing basic rotation invariance...")
    
    device = torch.device("cpu")  # Use CPU to avoid GPU issues
    
    # Create PIMEH model
    pimeh = PhysicsInformedMinimalEquivariantHead(hidden_dim=128)
    print(f"✅ PIMEH model created with {sum(p.numel() for p in pimeh.parameters())} parameters")
    
    # Create water molecule (H2O)
    angle_rad = 104.31 * math.pi / 180 / 2  # Half angle
    oh_distance = 0.9584
    
    positions = torch.tensor([
        [0.0, 0.0, 0.0],  # Oxygen at origin
        [oh_distance * math.cos(angle_rad), oh_distance * math.sin(angle_rad), 0.0],  # H1
        [oh_distance * math.cos(angle_rad), -oh_distance * math.sin(angle_rad), 0.0],  # H2
    ], dtype=torch.float32)
    
    # Random embeddings
    h = torch.randn(3, 128)
    batch = torch.tensor([0, 0, 0])
    
    print(f"✅ Test molecule created: H2O with {positions.shape[0]} atoms")
    
    # Test with identity (no rotation)
    pimeh.eval()
    with torch.no_grad():
        ref_constants = pimeh(h, positions, batch)
        print(f"✅ Reference constants: A={ref_constants[0,0]:.2f}, B={ref_constants[0,1]:.2f}, C={ref_constants[0,2]:.2f} GHz")
        
        # Test 90-degree rotation around Z-axis
        angle = math.pi / 2  # 90 degrees
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R_z = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        rotated_pos = positions @ R_z.T
        rot_constants = pimeh(h, rotated_pos, batch)
        
        # Calculate relative error
        rel_error = torch.abs(rot_constants - ref_constants) / (torch.abs(ref_constants) + 1e-10)
        max_error = rel_error.max().item()
        
        print(f"✅ After 90° Z rotation: A={rot_constants[0,0]:.2f}, B={rot_constants[0,1]:.2f}, C={rot_constants[0,2]:.2f} GHz")
        print(f"✅ Max relative error: {max_error:.2e}")
        
        # Test should pass with very small error
        tolerance = 1e-6
        if max_error < tolerance:
            print(f"✅ ROTATION INVARIANCE TEST PASSED (error {max_error:.2e} < {tolerance:.2e})")
            return True
        else:
            print(f"❌ ROTATION INVARIANCE TEST FAILED (error {max_error:.2e} >= {tolerance:.2e})")
            return False

def test_multiple_rotations():
    """Test multiple rotation scenarios.""" 
    print("\nTesting multiple rotations...")
    
    device = torch.device("cpu")
    pimeh = PhysicsInformedMinimalEquivariantHead(hidden_dim=128)
    
    # Simple diatomic molecule (HCl)
    positions = torch.tensor([
        [-0.5, 0.0, 0.0],  # H
        [0.5, 0.0, 0.0],   # Cl
    ], dtype=torch.float32)
    
    h = torch.randn(2, 128)
    batch = torch.tensor([0, 0])
    
    pimeh.eval()
    with torch.no_grad():
        ref_constants = pimeh(h, positions, batch)
        
        # Test different rotations
        rotations = [
            ("90° X", torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)),
            ("90° Y", torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=torch.float32)),
            ("180° Z", torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)),
        ]
        
        all_passed = True
        for name, R in rotations:
            rotated_pos = positions @ R.T
            rot_constants = pimeh(h, rotated_pos, batch)
            
            rel_error = torch.abs(rot_constants - ref_constants) / (torch.abs(ref_constants) + 1e-10)
            max_error = rel_error.max().item()
            
            if max_error < 1e-6:
                print(f"✅ {name}: error {max_error:.2e}")
            else:
                print(f"❌ {name}: error {max_error:.2e}")
                all_passed = False
        
        return all_passed

def test_edge_cases():
    """Test edge cases like single atoms."""
    print("\nTesting edge cases...")
    
    pimeh = PhysicsInformedMinimalEquivariantHead(hidden_dim=128)
    
    # Single atom
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    h = torch.randn(1, 128)
    batch = torch.tensor([0])
    
    pimeh.eval()
    with torch.no_grad():
        try:
            constants = pimeh(h, positions, batch)
            print(f"✅ Single atom test: A={constants[0,0]:.2f}, B={constants[0,1]:.2f}, C={constants[0,2]:.2f} GHz")
            return True
        except Exception as e:
            print(f"❌ Single atom test failed: {e}")
            return False

if __name__ == "__main__":
    print("ROTATION INVARIANCE VALIDATION")
    print("=" * 50)
    
    try:
        # Run basic tests
        test1 = test_basic_rotation_invariance()
        test2 = test_multiple_rotations()
        test3 = test_edge_cases()
        
        print("\n" + "=" * 50)
        if all([test1, test2, test3]):
            print("✅ ALL ROTATION INVARIANCE TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()