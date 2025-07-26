"""
Comprehensive Rotation Invariance Test Suite for PIMEH

This test suite validates that the Physics-Informed Minimal Equivariant Head (PIMEH)
produces rotation-invariant rotational constants, which is critical for physics correctness.
The tests cover:

1. PIMEH standalone rotation invariance 
2. Full DJMGNN with PIMEH rotation invariance
3. Various molecular structures (linear, planar, 3D)
4. Different rotation transformations (90°, 180°, random)
5. Batch processing and edge cases
6. Performance benchmarking

Key Requirements:
- Rotational constants A, B, C must be identical within numerical precision (<1e-6)
- SE(3) equivariance must hold for all molecular structures
- Tests must be efficient and well-documented
"""

import math
import time
from typing import Dict, List, Tuple, Optional, NamedTuple
import pytest

import torch
import torch.nn as nn
import numpy as np

# Project imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from moml.models.mgnn.pimeh import PhysicsInformedMinimalEquivariantHead, compute_inertia_tensor, ROTATIONAL_CONST_FACTOR_GHZ
from moml.models.mgnn.djmgnn import DJMGNN


class RotationTestResult(NamedTuple):
    """Results from a rotation invariance test."""
    passed: bool
    max_error: float
    mean_error: float
    test_name: str
    molecule_name: str
    num_rotations: int


class MolecularTestCase(NamedTuple):
    """Test case for a specific molecular structure."""
    name: str
    positions: torch.Tensor
    masses: torch.Tensor
    description: str
    expected_properties: Optional[Dict[str, float]] = None


# =====================================================================
# Rotation Matrix Utilities
# =====================================================================

def random_rotation_matrix(device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate a random 3D rotation matrix using Gram-Schmidt orthogonalization."""
    # Generate random matrix
    A = torch.randn(3, 3, device=device, dtype=dtype)
    
    # Gram-Schmidt orthogonalization
    Q, R = torch.linalg.qr(A)
    
    # Ensure proper rotation (det = +1)
    if torch.det(Q) < 0:
        Q[:, -1] *= -1
    
    return Q


def rotation_matrix_x(angle: float, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Rotation matrix around X-axis."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a]
    ], device=device, dtype=dtype)


def rotation_matrix_y(angle: float, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Rotation matrix around Y-axis.""" 
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [cos_a, 0.0, sin_a],
        [0.0, 1.0, 0.0],
        [-sin_a, 0.0, cos_a]
    ], device=device, dtype=dtype)


def rotation_matrix_z(angle: float, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Rotation matrix around Z-axis."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)


def euler_rotation_matrix(alpha: float, beta: float, gamma: float, 
                         device: torch.device = torch.device('cpu'), 
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create rotation matrix from Euler angles (ZYZ convention)."""
    Rz1 = rotation_matrix_z(alpha, device, dtype)
    Ry = rotation_matrix_y(beta, device, dtype)
    Rz2 = rotation_matrix_z(gamma, device, dtype)
    return Rz2 @ Ry @ Rz1


def get_standard_rotations(device: torch.device = torch.device('cpu'), 
                          dtype: torch.dtype = torch.float32) -> List[Tuple[str, torch.Tensor]]:
    """Get a set of standard test rotations."""
    pi = math.pi
    return [
        ("Identity", torch.eye(3, device=device, dtype=dtype)),
        ("90° X-axis", rotation_matrix_x(pi/2, device, dtype)),
        ("90° Y-axis", rotation_matrix_y(pi/2, device, dtype)),  
        ("90° Z-axis", rotation_matrix_z(pi/2, device, dtype)),
        ("180° X-axis", rotation_matrix_x(pi, device, dtype)),
        ("180° Y-axis", rotation_matrix_y(pi, device, dtype)),
        ("180° Z-axis", rotation_matrix_z(pi, device, dtype)),
        ("270° X-axis", rotation_matrix_x(3*pi/2, device, dtype)),
        ("270° Y-axis", rotation_matrix_y(3*pi/2, device, dtype)),
        ("270° Z-axis", rotation_matrix_z(3*pi/2, device, dtype)),
        ("45° X-axis", rotation_matrix_x(pi/4, device, dtype)),
        ("45° Y-axis", rotation_matrix_y(pi/4, device, dtype)), 
        ("45° Z-axis", rotation_matrix_z(pi/4, device, dtype)),
        ("Complex Euler (30°,60°,90°)", euler_rotation_matrix(pi/6, pi/3, pi/2, device, dtype)),
        ("Complex Euler (45°,90°,135°)", euler_rotation_matrix(pi/4, pi/2, 3*pi/4, device, dtype)),
    ]


# =====================================================================
# Molecular Test Case Generators  
# =====================================================================

def create_water_molecule(device: torch.device = torch.device('cpu'), 
                         dtype: torch.dtype = torch.float32) -> MolecularTestCase:
    """Create H2O test case with accurate geometry."""
    # NIST reference geometry: O-H = 0.9584 Å, H-O-H = 104.31°
    angle_rad = 104.31 * math.pi / 180 / 2  # Half angle
    oh_distance = 0.9584
    
    positions = torch.tensor([
        [0.0, 0.0, 0.0],  # Oxygen at origin
        [oh_distance * math.cos(angle_rad), oh_distance * math.sin(angle_rad), 0.0],  # H1
        [oh_distance * math.cos(angle_rad), -oh_distance * math.sin(angle_rad), 0.0],  # H2
    ], device=device, dtype=dtype)
    
    masses = torch.tensor([15.999, 1.008, 1.008], device=device, dtype=dtype)  # O, H, H
    
    return MolecularTestCase(
        name="H2O",
        positions=positions,
        masses=masses,
        description="Water molecule - bent geometry, asymmetric top",
        expected_properties={
            "A_lit": 835.1,  # Literature values in GHz
            "B_lit": 435.4,
            "C_lit": 278.1,
            "geometry": "bent",
            "symmetry": "C2v"
        }
    )


def create_co2_molecule(device: torch.device = torch.device('cpu'),
                       dtype: torch.dtype = torch.float32) -> MolecularTestCase:
    """Create CO2 test case - linear molecule."""
    # Linear CO2: O-C-O with C-O = 1.162 Å
    co_distance = 1.162
    
    positions = torch.tensor([
        [-co_distance, 0.0, 0.0],  # O1
        [0.0, 0.0, 0.0],           # C (center)
        [co_distance, 0.0, 0.0],   # O2
    ], device=device, dtype=dtype)
    
    masses = torch.tensor([15.999, 12.011, 15.999], device=device, dtype=dtype)  # O, C, O
    
    return MolecularTestCase(
        name="CO2",
        positions=positions,
        masses=masses,
        description="Carbon dioxide - linear molecule, special case",
        expected_properties={
            "geometry": "linear",
            "symmetry": "D∞h",
            "B_equals_C": True  # For linear molecules, B = C
        }
    )


def create_methane_molecule(device: torch.device = torch.device('cpu'),
                           dtype: torch.dtype = torch.float32) -> MolecularTestCase:
    """Create CH4 test case - tetrahedral geometry."""
    # Tetrahedral methane: C at center, H at tetrahedral positions
    ch_distance = 1.09
    
    # Tetrahedral coordinates (normalized, then scaled)
    sqrt3 = math.sqrt(3.0)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],                    # C at center
        [1.0, 1.0, 1.0],                   # H1
        [1.0, -1.0, -1.0],                 # H2
        [-1.0, 1.0, -1.0],                 # H3
        [-1.0, -1.0, 1.0],                 # H4
    ], device=device, dtype=dtype) * ch_distance / sqrt3
    
    masses = torch.tensor([12.011, 1.008, 1.008, 1.008, 1.008], device=device, dtype=dtype)  # C, H×4
    
    return MolecularTestCase(
        name="CH4",
        positions=positions,
        masses=masses,
        description="Methane - tetrahedral, spherical top",
        expected_properties={
            "geometry": "tetrahedral",
            "symmetry": "Td",
            "A_approx_B_approx_C": True  # Spherical top
        }
    )


def create_benzene_molecule(device: torch.device = torch.device('cpu'),
                           dtype: torch.dtype = torch.float32) -> MolecularTestCase:
    """Create C6H6 test case - planar ring geometry."""
    # Benzene ring: C-C = 1.39 Å, C-H = 1.08 Å
    cc_distance = 1.39
    ch_distance = 1.08
    
    positions = []
    masses = []
    
    # Create hexagonal ring of carbons
    for i in range(6):
        angle = i * math.pi / 3  # 60° increments
        x = cc_distance * math.cos(angle)
        y = cc_distance * math.sin(angle) 
        z = 0.0
        positions.append([x, y, z])
        masses.append(12.011)  # Carbon mass
        
        # Add hydrogen for each carbon
        h_x = (cc_distance + ch_distance) * math.cos(angle)
        h_y = (cc_distance + ch_distance) * math.sin(angle)
        h_z = 0.0
        positions.append([h_x, h_y, h_z])
        masses.append(1.008)  # Hydrogen mass
    
    positions = torch.tensor(positions, device=device, dtype=dtype)
    masses = torch.tensor(masses, device=device, dtype=dtype)
    
    return MolecularTestCase(
        name="C6H6",
        positions=positions,
        masses=masses,
        description="Benzene - planar ring, symmetric top",
        expected_properties={
            "geometry": "planar",
            "symmetry": "D6h",
            "planar": True
        }
    )


def create_linear_diatomic(element1: str = "H", element2: str = "Cl", 
                          bond_length: float = 1.27,
                          device: torch.device = torch.device('cpu'),
                          dtype: torch.dtype = torch.float32) -> MolecularTestCase:
    """Create a simple diatomic molecule."""
    # Atomic masses (approximate)
    atomic_masses = {
        "H": 1.008, "He": 4.003, "Li": 6.941, "C": 12.011, "N": 14.007,
        "O": 15.999, "F": 18.998, "Cl": 35.453, "Br": 79.904, "I": 126.90
    }
    
    positions = torch.tensor([
        [-bond_length/2, 0.0, 0.0],  # Atom 1
        [bond_length/2, 0.0, 0.0],   # Atom 2
    ], device=device, dtype=dtype)
    
    masses = torch.tensor([
        atomic_masses.get(element1, 1.0),
        atomic_masses.get(element2, 1.0)
    ], device=device, dtype=dtype)
    
    return MolecularTestCase(
        name=f"{element1}{element2}",
        positions=positions,
        masses=masses,
        description=f"{element1}-{element2} diatomic molecule - linear",
        expected_properties={
            "geometry": "linear",
            "diatomic": True
        }
    )


def create_single_atom(element: str = "Ar", 
                      device: torch.device = torch.device('cpu'),
                      dtype: torch.dtype = torch.float32) -> MolecularTestCase:
    """Create single atom test case (edge case)."""
    atomic_masses = {"H": 1.008, "He": 4.003, "Ar": 39.948, "Kr": 83.798}
    
    positions = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)
    masses = torch.tensor([atomic_masses.get(element, 1.0)], device=device, dtype=dtype)
    
    return MolecularTestCase(
        name=element,
        positions=positions,
        masses=masses,
        description=f"Single {element} atom - edge case",
        expected_properties={
            "geometry": "atomic",
            "single_atom": True
        }
    )


def get_all_molecular_test_cases(device: torch.device = torch.device('cpu'),
                                dtype: torch.dtype = torch.float32) -> List[MolecularTestCase]:
    """Get comprehensive set of molecular test cases."""
    return [
        create_water_molecule(device, dtype),
        create_co2_molecule(device, dtype), 
        create_methane_molecule(device, dtype),
        create_benzene_molecule(device, dtype),
        create_linear_diatomic("H", "Cl", 1.27, device, dtype),
        create_linear_diatomic("N", "N", 1.10, device, dtype),  # N2
        create_linear_diatomic("O", "O", 1.21, device, dtype),  # O2
        create_single_atom("Ar", device, dtype),
        create_single_atom("Kr", device, dtype),
    ]


# =====================================================================
# Core Test Functions
# =====================================================================

def _run_rotation_invariance_test(model: nn.Module,
                                h: torch.Tensor,
                                pos: torch.Tensor,
                                batch: torch.Tensor,
                                rotations: List[Tuple[str, torch.Tensor]],
                                tolerance: float = 1e-6,
                                test_name: str = "generic",
                                edge_index: Optional[torch.Tensor] = None,
                                dist: Optional[torch.Tensor] = None) -> RotationTestResult:
    """
    Test rotation invariance for a given model and molecular data.
    
    Args:
        model: PIMEH or DJMGNN model to test
        h: Node embeddings [N, hidden_dim]
        pos: Atomic positions [N, 3]
        batch: Batch indices [N]
        rotations: List of (name, rotation_matrix) pairs
        tolerance: Maximum allowed relative error
        test_name: Name for this test
        
    Returns:
        RotationTestResult with test results
    """
    model.eval()
    
    with torch.no_grad():
        # Get reference rotational constants
        if hasattr(model, 'physics_head'):  # DJMGNN case
            output = model(x=h, edge_index=edge_index, edge_attr=None, batch=batch, pos=pos, dist=dist)
            ref_constants = output['graph_pred'][:, 16:19]  # Rotational constants columns
        else:  # PIMEH case
            ref_constants = model(h, pos, batch)
        
        errors = []
        max_error = 0.0
        
        for rotation_name, R in rotations:
            # Apply rotation to positions
            rotated_pos = pos @ R.T  # [N, 3] @ [3, 3] -> [N, 3]
            
            # Get rotational constants for rotated positions
            if hasattr(model, 'physics_head'):  # DJMGNN case
                output = model(x=h, edge_index=edge_index, edge_attr=None, batch=batch, pos=rotated_pos, dist=dist)
                rot_constants = output['graph_pred'][:, 16:19]
            else:  # PIMEH case
                rot_constants = model(h, rotated_pos, batch)
            
            # Calculate relative error
            rel_error = torch.abs(rot_constants - ref_constants) / (torch.abs(ref_constants) + 1e-10)
            error = rel_error.max().item()
            errors.append(error)
            max_error = max(max_error, error)
            
            if error > tolerance:
                print(f"❌ Rotation {rotation_name}: relative error {error:.2e} > {tolerance:.2e}")
                return RotationTestResult(
                    passed=False,
                    max_error=max_error,
                    mean_error=np.mean(errors),
                    test_name=test_name,
                    molecule_name="unknown",
                    num_rotations=len(rotations)
                )
        
        mean_error = np.mean(errors)
        return RotationTestResult(
            passed=True,
            max_error=max_error,
            mean_error=mean_error,
            test_name=test_name,
            molecule_name="unknown",
            num_rotations=len(rotations)
        )


# =====================================================================
# Test Classes
# =====================================================================

class TestRotationInvariance:
    """Comprehensive rotation invariance test suite."""
    
    @pytest.fixture
    def device(self):
        """Get appropriate device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def tolerance(self):
        """Numerical tolerance for rotation invariance."""
        return 1e-6
    
    @pytest.fixture
    def pimeh_model(self, device):
        """Create PIMEH model for testing."""
        model = PhysicsInformedMinimalEquivariantHead(hidden_dim=128)
        return model.to(device)
    
    @pytest.fixture
    def djmgnn_model(self, device):
        """Create DJMGNN model with PIMEH for testing."""
        model = DJMGNN(
            in_node_dim=33,  # Updated to match config
            hidden_dim=160,
            n_blocks=4,
            layers_per_block=2,
            graph_output_dims=19,
            node_output_dims=3,
            energy_output_dims=1,
            dropout=0.2,
            jk_mode="attention",
            use_supernode=True,
            use_rbf=True,
            rbf_K=32
        )
        return model.to(device)

    @pytest.mark.parametrize("molecule_case", [
        "H2O", "CO2", "CH4", "C6H6", "HCl", "N2", "Ar"
    ])
    def test_pimeh_rotation_invariance(self, pimeh_model, device, tolerance, molecule_case):
        """Test PIMEH rotation invariance for different molecular structures."""
        # Get molecular test case
        test_cases = {case.name: case for case in get_all_molecular_test_cases(device)}
        
        if molecule_case not in test_cases:
            pytest.skip(f"Molecule {molecule_case} not available")
            
        mol_case = test_cases[molecule_case]
        
        # Create test data
        n_atoms = mol_case.positions.size(0)
        h = torch.randn(n_atoms, 128, device=device)  # Random embeddings
        batch = torch.zeros(n_atoms, dtype=torch.long, device=device)  # Single molecule
        
        # Get standard rotations
        rotations = get_standard_rotations(device)
        
        # Test rotation invariance
        result = _run_rotation_invariance_test(
            pimeh_model, h, mol_case.positions, batch, rotations, tolerance,
            f"PIMEH_{molecule_case}"
        )
        
        assert result.passed, f"PIMEH rotation invariance failed for {molecule_case}: max_error={result.max_error:.2e}"
        print(f"✅ PIMEH {molecule_case}: max_error={result.max_error:.2e}, mean_error={result.mean_error:.2e}")

    @pytest.mark.parametrize("molecule_case", [
        "H2O", "CO2", "CH4", "HCl"  # Smaller set for full DJMGNN tests
    ])
    def test_djmgnn_rotation_invariance(self, djmgnn_model, device, tolerance, molecule_case):
        """Test full DJMGNN with PIMEH rotation invariance."""
        from torch_geometric.data import Data
        from moml.data.feature_transforms import CreateEdges

        # Get molecular test case
        test_cases = {case.name: case for case in get_all_molecular_test_cases(device)}

        if molecule_case not in test_cases:
            pytest.skip(f"Molecule {molecule_case} not available")

        mol_case = test_cases[molecule_case]

        # Create full molecular graph data
        n_atoms = mol_case.positions.size(0)
        h = torch.randn(n_atoms, 33, device=device)  # Full node features (33 dim)
        batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

        # Create graph structure
        data = Data(pos=mol_case.positions)
        data = CreateEdges(cutoff=5.0)(data)
        edge_index = data.edge_index.to(device)
        dist = torch.norm(data.pos[edge_index[0]] - data.pos[edge_index[1]], p=2, dim=1).unsqueeze(-1).to(device)

        # Get subset of rotations (DJMGNN is slower)
        standard_rotations = get_standard_rotations(device)
        key_rotations = [
            standard_rotations[0],  # Identity
            standard_rotations[1],  # 90° X
            standard_rotations[5],  # 180° Y
            standard_rotations[9],  # 270° Z
            standard_rotations[13], # Complex Euler
        ]

        # Test rotation invariance
        result = _run_rotation_invariance_test(
            djmgnn_model, h, mol_case.positions, batch, key_rotations, tolerance,
            f"DJMGNN_{molecule_case}", edge_index=edge_index, dist=dist
        )

        assert result.passed, f"DJMGNN rotation invariance failed for {molecule_case}: max_error={result.max_error:.2e}"
        print(f"✅ DJMGNN {molecule_case}: max_error={result.max_error:.2e}, mean_error={result.mean_error:.2e}")

    def test_random_rotations(self, pimeh_model, device, tolerance):
        """Test with many random rotations."""
        # Use water molecule
        water = create_water_molecule(device)
        n_atoms = water.positions.size(0)
        h = torch.randn(n_atoms, 128, device=device)
        batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
        
        # Generate random rotations
        num_random = 50
        random_rotations = []
        for i in range(num_random):
            R = random_rotation_matrix(device)
            random_rotations.append((f"Random_{i+1}", R))
        
        result = _run_rotation_invariance_test(
            pimeh_model, h, water.positions, batch, random_rotations, tolerance,
            "PIMEH_Random_Rotations"
        )
        
        assert result.passed, f"Random rotation test failed: max_error={result.max_error:.2e}"
        print(f"✅ Random rotations: {num_random} tests, max_error={result.max_error:.2e}")

    def test_batch_processing(self, pimeh_model, device, tolerance):
        """Test rotation invariance with batch processing."""
        # Create batch with multiple molecules
        molecules = [
            create_water_molecule(device),
            create_co2_molecule(device),
            create_methane_molecule(device)
        ]
        
        # Combine into batch
        positions = []
        embeddings = []
        batch_indices = []
        
        for mol_id, mol in enumerate(molecules):
            n_atoms = mol.positions.size(0)
            positions.append(mol.positions)
            embeddings.append(torch.randn(n_atoms, 128, device=device))
            batch_indices.extend([mol_id] * n_atoms)
        
        pos = torch.cat(positions, dim=0)
        h = torch.cat(embeddings, dim=0)
        batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
        
        # Test with key rotations
        key_rotations = get_standard_rotations(device)[:5]  # First 5 rotations
        
        result = _run_rotation_invariance_test(
            pimeh_model, h, pos, batch, key_rotations, tolerance,
            "PIMEH_Batch_Processing"
        )
        
        assert result.passed, f"Batch rotation test failed: max_error={result.max_error:.2e}"
        print(f"✅ Batch processing: 3 molecules, max_error={result.max_error:.2e}")

    def test_edge_cases(self, pimeh_model, device, tolerance):
        """Test edge cases: single atoms, linear molecules."""
        # Single atom
        single_atom = create_single_atom("Ar", device)
        h_single = torch.randn(1, 128, device=device)
        batch_single = torch.zeros(1, dtype=torch.long, device=device)
        
        # Test single atom - should handle gracefully
        rotations = get_standard_rotations(device)[:3]  # Just a few rotations
        result = _run_rotation_invariance_test(
            pimeh_model, h_single, single_atom.positions, batch_single,
            rotations, tolerance, "PIMEH_Single_Atom"
        )
        
        # Single atoms might have special handling, so we just check it doesn't crash
        print(f"✅ Single atom test: passed={result.passed}, max_error={result.max_error:.2e}")
        
        # Linear molecule (CO2)
        co2 = create_co2_molecule(device)
        n_atoms = co2.positions.size(0)
        h_linear = torch.randn(n_atoms, 128, device=device)
        batch_linear = torch.zeros(n_atoms, dtype=torch.long, device=device)
        
        result = _run_rotation_invariance_test(
            pimeh_model, h_linear, co2.positions, batch_linear,
            rotations, tolerance, "PIMEH_Linear_CO2"
        )
        
        assert result.passed, f"Linear molecule test failed: max_error={result.max_error:.2e}"
        print(f"✅ Linear molecule: max_error={result.max_error:.2e}")


# =====================================================================
# Performance Benchmarking
# =====================================================================

def benchmark_rotation_invariance(model: nn.Module,
                                 batch_sizes: List[int] = [1, 4, 8, 16],
                                 num_rotations: int = 10,
                                 device: torch.device = torch.device('cpu')) -> Dict[str, float]:
    """Benchmark rotation invariance testing performance."""
    print(f"\n{'='*60}")
    print("ROTATION INVARIANCE PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create batch data with water molecules
        molecules = [create_water_molecule(device) for _ in range(batch_size)]
        
        positions = []
        embeddings = []
        batch_indices = []
        
        for mol_id, mol in enumerate(molecules):
            n_atoms = mol.positions.size(0)
            positions.append(mol.positions)
            
            if hasattr(model, 'physics_head'):  # DJMGNN
                embeddings.append(torch.randn(n_atoms, 29, device=device))
            else:  # PIMEH
                embeddings.append(torch.randn(n_atoms, 128, device=device))
                
            batch_indices.extend([mol_id] * n_atoms)
        
        pos = torch.cat(positions, dim=0)
        h = torch.cat(embeddings, dim=0)
        batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
        
        # Generate random rotations
        rotations = [(f"R{i}", random_rotation_matrix(device)) for i in range(num_rotations)]
        
        # Benchmark
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            from torch_geometric.data import Data
            from moml.data.feature_transforms import CreateEdges
            edge_index = CreateEdges(cutoff=5.0)(Data(pos=pos)).edge_index.to(device)
            dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], p=2, dim=1).unsqueeze(-1).to(device)
            result = _run_rotation_invariance_test(
                model, h, pos, batch, rotations, 1e-6, f"Benchmark_B{batch_size}",
                edge_index=edge_index, dist=dist
            )
        
        elapsed = time.time() - start_time
        tests_per_sec = (batch_size * num_rotations) / elapsed
        
        print(f"Batch Size {batch_size:2d}: {elapsed:.3f}s total, "
              f"{tests_per_sec:.1f} tests/sec, passed={result.passed}")
        
        results[f"batch_{batch_size}"] = {
            "elapsed_time": elapsed,
            "tests_per_second": tests_per_sec,
            "passed": result.passed,
            "max_error": result.max_error
        }
    
    return results


# =====================================================================
# Main Test Execution
# =====================================================================

if __name__ == "__main__":
    print("COMPREHENSIVE ROTATION INVARIANCE TEST SUITE")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test PIMEH standalone
    print("\n🧪 Testing PIMEH standalone...")
    pimeh = PhysicsInformedMinimalEquivariantHead(hidden_dim=128).to(device)
    
    # Test all molecular structures
    test_cases = get_all_molecular_test_cases(device)
    rotations = get_standard_rotations(device)
    
    for mol_case in test_cases[:4]:  # Test first 4 molecules
        n_atoms = mol_case.positions.size(0)
        h = torch.randn(n_atoms, 128, device=device)
        batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
        
        result = _run_rotation_invariance_test(
            pimeh, h, mol_case.positions, batch, rotations, 1e-6,
            f"PIMEH_{mol_case.name}"
        )
        
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} {mol_case.name}: max_error={result.max_error:.2e}")
    
    # Benchmark performance
    print("\n⚡ Benchmarking PIMEH performance...")
    pimeh_results = benchmark_rotation_invariance(pimeh, [1, 4, 8], 5, device)
    
    print(f"\n{'='*60}")
    print("✅ ROTATION INVARIANCE TEST SUITE COMPLETED")
    print(f"{'='*60}")