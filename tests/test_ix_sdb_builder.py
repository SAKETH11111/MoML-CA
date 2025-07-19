"""
tests/test_ix_sdb_builder.py

Unit tests for the IX-SDB bead builder plugin in
moml.simulation.molecular_dynamics.force_field.plugins.ix_sdb_v1.build.
"""

import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest

from moml.simulation.molecular_dynamics.force_field.plugins.ix_sdb_v1.build import (
    build, _pack_bead
)

try:
    import openmm.unit as unit
except ImportError:
    unit = None

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
    """
    Test suite for IX-SDB bead builder functionality.
    """

    @pytest.fixture
    def test_monomer(self) -> Any:
        """
        Create a test monomer molecule.

        Returns:
            Any: The test monomer molecule.
        """
        if not OPENFF_AVAILABLE or Molecule is None:
            pytest.skip("OpenFF toolkit not available")
        return Molecule.from_smiles("C[N+](C)(C)Cc1ccc(C=C)cc1.[Cl-]")

    @pytest.fixture
    def small_monomer(self) -> Any:
        """
        Create a very simple monomer for faster testing.

        Returns:
            Any: The small monomer molecule.
        """
        if not OPENFF_AVAILABLE or Molecule is None:
            pytest.skip("OpenFF toolkit not available")
        return Molecule.from_smiles("CCO")

    def test_module_imports(self) -> None:
        """
        Test that the module can be imported regardless of dependencies.
        """
        assert hasattr(build, '__call__')
        assert hasattr(_pack_bead, '__call__')

    def test_build_requires_openff(self) -> None:
        """
        Test that build function properly handles missing OpenFF.
        """
        if OPENFF_AVAILABLE:
            pytest.skip("OpenFF is available, can't test missing dependency handling")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = {"bead_count": 2, "bead_radius_nm": 1.0}
            with pytest.raises(ImportError, match="openff-toolkit is required"):
                build(tmp_path, config)

    def test_pack_bead_requires_rdkit(self) -> None:
        """
        Test that _pack_bead properly handles missing RDKit.
        """
        if not OPENFF_AVAILABLE or Molecule is None:
            pytest.skip("OpenFF toolkit not available for this test")
        if RDKIT_AVAILABLE:
            pytest.skip("RDKit is available, can't test missing dependency handling")
        monomer = Molecule.from_smiles("CCO") if Molecule is not None else None
        if monomer is None:
            pytest.skip("Molecule class not available")
        with pytest.raises(ImportError, match="rdkit is required"):
            _pack_bead(monomer, 1.0, 2)

    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, reason="OpenFF toolkit and RDKit required")
    def test_pack_bead_basic_functionality(self, small_monomer: Any) -> None:
        """
        Test that _pack_bead creates non-empty topology and positions.
        """
        radius_nm = 1.0
        count = 3
        positions, topology = _pack_bead(small_monomer, radius_nm, count)
        assert len(positions) > 0, "No positions generated"
        assert topology.getNumAtoms() > 0, "Empty topology created"
        expected_atoms_per_mol = 9
        expected_total = count * expected_atoms_per_mol
        actual_atoms = topology.getNumAtoms()
        assert actual_atoms >= count * 3, f"Too few atoms: {actual_atoms} < {count * 3}"
        assert actual_atoms <= count * 15, f"Too many atoms: {actual_atoms} > {count * 15}"

    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, reason="OpenFF toolkit and RDKit required")
    def test_pack_bead_positions_within_sphere(self, small_monomer: Any) -> None:
        """
        Test that all positions are within the specified sphere.
        """
        radius_nm = 2.0
        count = 2
        positions, _ = _pack_bead(small_monomer, radius_nm, count)
        for pos in positions:
            distance = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
            assert distance <= radius_nm * 1.1, f"Position outside sphere: {distance} > {radius_nm}"

    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, reason="OpenFF toolkit and RDKit required")
    def test_pack_bead_with_real_monomer(self, test_monomer: Any) -> None:
        """
        Test with the actual IX-SDB monomer structure.
        """
        radius_nm = 1.5
        count = 2
        positions, topology = _pack_bead(test_monomer, radius_nm, count)
        assert len(positions) > 0
        assert topology.getNumAtoms() > 0
        expected_min = count * 20
        expected_max = count * 60
        actual_atoms = topology.getNumAtoms()
        assert expected_min <= actual_atoms <= expected_max, (
            f"Unexpected atom count: {actual_atoms} not in [{expected_min}, {expected_max}]"
        )

    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, reason="OpenFF toolkit and RDKit required")
    def test_build_function_integration(self) -> None:
        """
        Test the main build function that uses _pack_bead.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = {"bead_count": 2, "bead_radius_nm": 1.0}
            pdb_path, topology, atom_indices = build(tmp_path, config)
            assert pdb_path.exists()
            assert pdb_path.suffix == '.pdb'
            assert topology.getNumAtoms() > 0
            assert len(atom_indices) == topology.getNumAtoms()
            assert all(isinstance(idx, int) for idx in atom_indices)

    @pytest.mark.skipif(not OPENFF_AVAILABLE or not RDKIT_AVAILABLE, reason="OpenFF toolkit and RDKit required")
    def test_larger_bead_scaling(self, small_monomer: Any) -> None:
        """
        Test that larger beads can be created and positions are reasonable.
        """
        radius_nm = 2.5
        count = 20
        positions, topology = _pack_bead(small_monomer, radius_nm, count)
        assert len(positions) > 0
        assert topology.getNumAtoms() > 0
        if unit is None:
            pytest.skip("openmm.unit not available")
        try:
            nanometer_unit = unit.nanometer  # type: ignore[attr-defined]
        except AttributeError:
            pytest.skip("openmm.unit.nanometer not available")
        for pos in positions:
            distance = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
            distance_nm = distance.value_in_unit(nanometer_unit) if hasattr(distance, 'value_in_unit') else distance
            assert distance_nm <= radius_nm * 1.2, f"Position too far from center: {distance_nm} > {radius_nm * 1.2}"
        expected_min_atoms = min(count, 10) * 3
        assert topology.getNumAtoms() >= expected_min_atoms
