"""
Unit tests for the IX-SDB bead builder plugin in
moml.simulation.molecular_dynamics.force_field.plugins.ix_sdb_v1.build.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from moml.simulation.molecular_dynamics.force_field.plugins.ix_sdb_v1.build import build, _pack_bead

# Import OpenMM unit for the extended test
try:
    import openmm.unit as unit
except ImportError:
    unit = None

# Try to import optional dependencies
try:
    from openff.toolkit.topology import Molecule
    OPENFF_AVAILABLE = True
except ImportError:
    OPENFF_AVAILABLE = False
    Molecule = None

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class TestIXSDBBuilder:
    """Test suite for IX-SDB bead builder functionality."""
    
    @pytest.fixture
    def test_monomer(self):
        """Create a test monomer molecule."""
        if not OPENFF_AVAILABLE:
            pytest.skip("OpenFF toolkit not available")
        # Simplified monomer for testing - quaternary ammonium styrene with chloride
        return Molecule.from_smiles("C[N+](C)(C)Cc1ccc(C=C)cc1.[Cl-]")
    
    @pytest.fixture
    def small_monomer(self):
        """Create a very simple monomer for faster testing."""
        if not OPENFF_AVAILABLE:
            pytest.skip("OpenFF toolkit not available")
        # Even simpler molecule for quick tests
        return Molecule.from_smiles("CCO")  # Ethanol
    
    def test_module_imports(self):
        """Test that the module can be imported regardless of dependencies."""
        # This test should always pass
        assert hasattr(build, '__call__')
        assert hasattr(_pack_bead, '__call__')
    
    def test_build_requires_openff(self):
        """Test that build function properly handles missing OpenFF."""
        if OPENFF_AVAILABLE:
            pytest.skip("OpenFF is available, can't test missing dependency handling")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = {"bead_count": 2, "bead_radius_nm": 1.0}
            
            with pytest.raises(ImportError, match="openff-toolkit is required"):
                build(tmp_path, config)
    
    def test_pack_bead_requires_rdkit(self):
        """Test that _pack_bead properly handles missing RDKit."""
        if not OPENFF_AVAILABLE:
            pytest.skip("OpenFF toolkit not available for this test")
        if RDKIT_AVAILABLE:
            pytest.skip("RDKit is available, can't test missing dependency handling")
        
        monomer = Molecule.from_smiles("CCO")
        with pytest.raises(ImportError, match="rdkit is required"):
            _pack_bead(monomer, 1.0, 2)
    
    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, 
                        reason="OpenFF toolkit and RDKit required")
    def test_pack_bead_basic_functionality(self, small_monomer):
        """Test that _pack_bead creates non-empty topology and positions."""
        radius_nm = 1.0
        count = 3
        
        positions, topology = _pack_bead(small_monomer, radius_nm, count)
        
        # Should have created positions and topology
        assert len(positions) > 0, "No positions generated"
        assert topology.getNumAtoms() > 0, "Empty topology created"
        
        # Should have approximately the right number of atoms
        # Each ethanol molecule has ~9 atoms (C-C-O + hydrogens)
        expected_atoms_per_mol = 9  # rough estimate for ethanol with hydrogens
        expected_total = count * expected_atoms_per_mol
        actual_atoms = topology.getNumAtoms()
        
        # Allow some flexibility in atom count due to different H addition strategies
        assert actual_atoms >= count * 3, f"Too few atoms: {actual_atoms} < {count * 3}"
        assert actual_atoms <= count * 15, f"Too many atoms: {actual_atoms} > {count * 15}"
    
    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, 
                        reason="OpenFF toolkit and RDKit required")
    def test_pack_bead_positions_within_sphere(self, small_monomer):
        """Test that all positions are within the specified sphere."""
        radius_nm = 2.0
        count = 2
        
        positions, topology = _pack_bead(small_monomer, radius_nm, count)
        
        # Check that all positions are within the sphere
        for pos in positions:
            distance = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
            assert distance <= radius_nm * 1.1, f"Position outside sphere: {distance} > {radius_nm}"
    
    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, 
                        reason="OpenFF toolkit and RDKit required")
    def test_pack_bead_with_real_monomer(self, test_monomer):
        """Test with the actual IX-SDB monomer structure."""
        radius_nm = 1.5
        count = 2  # Small count for faster testing
        
        positions, topology = _pack_bead(test_monomer, radius_nm, count)
        
        # Should successfully pack the real monomer
        assert len(positions) > 0
        assert topology.getNumAtoms() > 0
        
        # Real monomer should have more atoms (quaternary ammonium + aromatic ring)
        # Estimate ~30-40 atoms per monomer with hydrogens
        expected_min = count * 20
        expected_max = count * 60
        actual_atoms = topology.getNumAtoms()
        
        assert expected_min <= actual_atoms <= expected_max, \
            f"Unexpected atom count: {actual_atoms} not in [{expected_min}, {expected_max}]"
    
    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, 
                        reason="OpenFF toolkit and RDKit required")
    def test_build_function_integration(self):
        """Test the main build function that uses _pack_bead."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            config = {
                "bead_count": 2,
                "bead_radius_nm": 1.0
            }
            
            pdb_path, topology, atom_indices = build(tmp_path, config)
            
            # Should create a PDB file
            assert pdb_path.exists()
            assert pdb_path.suffix == '.pdb'
            
            # Should have valid topology and atom indices
            assert topology.getNumAtoms() > 0
            assert len(atom_indices) == topology.getNumAtoms()
            assert all(isinstance(idx, int) for idx in atom_indices)
    
    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, 
                        reason="OpenFF toolkit and RDKit required")
    def test_larger_bead_scaling(self, small_monomer):
        """Test that larger beads can be created and positions are reasonable."""
        radius_nm = 2.5
        count = 20  # Larger bead to test performance
        
        positions, topology = _pack_bead(small_monomer, radius_nm, count)
        
        # Should successfully create a larger bead
        assert len(positions) > 0
        assert topology.getNumAtoms() > 0
        
        # All positions should be within the sphere (with small tolerance)
        for pos in positions:
            distance = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
            # Convert units properly - pos should have nanometer units
            distance_nm = distance.value_in_unit(unit.nanometer) if hasattr(distance, 'value_in_unit') else distance
            assert distance_nm <= radius_nm * 1.2, f"Position too far from center: {distance_nm} > {radius_nm * 1.2}"
        
        # Should place a reasonable number of monomers
        expected_min_atoms = min(count, 10) * 3  # At least some monomers placed
        assert topology.getNumAtoms() >= expected_min_atoms