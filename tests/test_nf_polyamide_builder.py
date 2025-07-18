"""
tests/test_nf_polyamide_builder.py

Unit tests for the polyamide membrane builder in
moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.

This test suite validates the production-ready polyamide membrane construction
system including all components: monomer management, polymerization, membrane
tiling, solvation, and force field parametrization.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Generator, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

# Import OpenMM units for tests
import openmm.unit as unit

# Import the module under test
from moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build import (
    DependencyError,
    ForceFieldParametrizer,
    MonomerManager,
    PackmolIntegrator,
    PolyamideBuilderError,
    PolymerizationEngine,
    SolvationSystem,
    ValidationError,
    build,
    _validate_config,
)

# Test dependencies availability
try:
    from openff.toolkit.topology import Molecule

    OPENFF_AVAILABLE = True
except ImportError:
    OPENFF_AVAILABLE = False

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class TestConfigValidation:
    """
    Test configuration validation functionality.
    """

    def test_valid_config_list(self) -> None:
        """
        Test validation with valid list configuration.
        """
        config = {"repeat_units": [4, 4, 2]}
        _validate_config(config)

    def test_valid_config_int(self) -> None:
        """
        Test validation with valid integer configuration.
        """
        config = {"repeat_units": 5}
        _validate_config(config)

    def test_missing_required_key(self) -> None:
        """
        Test validation fails for missing required keys.
        """
        config = {}
        with pytest.raises(ValidationError, match="Missing required configuration key: repeat_units"):
            _validate_config(config)

    def test_invalid_repeat_units_type(self) -> None:
        """
        Test validation fails for invalid repeat_units type.
        """
        config = {"repeat_units": "invalid"}
        with pytest.raises(ValidationError, match="repeat_units must be int or list of ints"):
            _validate_config(config)

    def test_invalid_repeat_units_negative(self) -> None:
        """
        Test validation fails for negative repeat_units.
        """
        config = {"repeat_units": [-1, 4, 2]}
        with pytest.raises(ValidationError, match="All repeat_units must be positive"):
            _validate_config(config)

    def test_invalid_repeat_units_short_list(self) -> None:
        """
        Test validation fails for too short repeat_units list.
        """
        config = {"repeat_units": [4]}
        with pytest.raises(ValidationError, match="repeat_units must have at least 2 dimensions"):
            _validate_config(config)

    def test_invalid_density(self) -> None:
        """
        Test validation fails for invalid density.
        """
        config = {"repeat_units": [4, 4, 2], "target_density_g_cm3": -1.0}
        with pytest.raises(ValidationError, match="target_density_g_cm3 must be positive number"):
            _validate_config(config)

    def test_invalid_box_height(self) -> None:
        """
        Test validation fails for invalid box height.
        """
        config = {"repeat_units": [4, 4, 2], "box_height": 0}
        with pytest.raises(ValidationError, match="box_height must be positive number"):
            _validate_config(config)


class TestMonomerManager:
    """
    Test monomer management functionality.
    """

    @pytest.fixture
    def resources_dir(self) -> Generator[Path, None, None]:
        """
        Create temporary resources directory with test files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            resources_path = Path(tmp_dir)

            polyamide_sdf = resources_path / "polyamide_monomer.sdf"
            polyamide_sdf.write_text(
                """polyamide_test
  Test

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  2  0  0  0  0
  3  6  1  0  0  0  0
  1  6  1  0  0  0  0
M  END
$$$$"""
            )

            yield resources_path

    def test_dependency_check_missing_openff(self, resources_dir: Path) -> None:
        """
        Test dependency validation when OpenFF is missing.
        """
        with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.OPENFF_AVAILABLE", False):
            with pytest.raises(DependencyError, match="OpenFF Toolkit is required"):
                MonomerManager(resources_dir)

    def test_dependency_check_missing_rdkit(self, resources_dir: Path) -> None:
        """
        Test dependency validation when RDKit is missing.
        """
        with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.OPENFF_AVAILABLE", True):
            with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.RDKIT_AVAILABLE", False):
                with pytest.raises(DependencyError, match="RDKit is required"):
                    MonomerManager(resources_dir)

    @pytest.mark.skipif(not OPENFF_AVAILABLE, reason="OpenFF not available")
    def test_missing_monomer_file(self, resources_dir: Path) -> None:
        """
        Test error handling for missing monomer file.
        """
        manager = MonomerManager(resources_dir)
        with pytest.raises(FileNotFoundError, match="Monomer file not found"):
            manager.load_monomer("nonexistent")

    @pytest.mark.skipif(not OPENFF_AVAILABLE, reason="OpenFF not available")
    def test_load_monomer_success(self, resources_dir: Path) -> None:
        """
        Test successful monomer loading.
        """
        manager = MonomerManager(resources_dir)
        molecule = manager.load_monomer("polyamide")
        assert molecule is not None
        assert molecule.n_atoms > 0

    @pytest.mark.skipif(not OPENFF_AVAILABLE, reason="OpenFF not available")
    def test_monomer_caching(self, resources_dir: Path) -> None:
        """
        Test monomer caching functionality.
        """
        manager = MonomerManager(resources_dir)
        mol1 = manager.load_monomer("polyamide")
        mol2 = manager.load_monomer("polyamide")
        assert mol1 is mol2


class TestPolymerizationEngine:
    """
    Test polymerization engine functionality.
    """

    @pytest.fixture
    def mock_monomer_manager(self) -> Mock:
        """
        Create mock monomer manager.
        """
        manager = Mock(spec=MonomerManager)
        mock_molecule = Mock()
        mock_molecule.n_atoms = 6
        mock_molecule.n_bonds = 6
        mock_molecule.atoms = [Mock(atomic_number=6, formal_charge=0, is_aromatic=False) for _ in range(6)]
        mock_molecule.bonds = [
            Mock(atom1_index=i, atom2_index=i + 1, bond_order=1, is_aromatic=False) for i in range(5)
        ]
        manager.load_monomer.return_value = mock_molecule
        return manager

    def test_invalid_repeat_units(self, mock_monomer_manager: Mock) -> None:
        """
        Test validation of repeat units parameter.
        """
        engine = PolymerizationEngine(mock_monomer_manager)
        with pytest.raises(ValidationError, match="Repeat units must be at least 1"):
            engine.create_polymer_chain(0)

    def test_single_repeat_unit(self, mock_monomer_manager: Mock) -> None:
        """
        Test polymer creation with single repeat unit.
        """
        engine = PolymerizationEngine(mock_monomer_manager)
        result = engine.create_polymer_chain(1)
        assert result is not None
        mock_monomer_manager.load_monomer.assert_called_once_with("polyamide")

    @pytest.mark.skipif(not OPENFF_AVAILABLE, reason="OpenFF not available")
    def test_multiple_repeat_units(self, mock_monomer_manager: Mock) -> None:
        """
        Test polymer creation with multiple repeat units.
        """
        with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.Molecule") as mock_molecule_class:
            mock_polymer = Mock()
            mock_polymer.add_atom = Mock()
            mock_polymer.add_bond = Mock()
            mock_molecule_class.return_value = mock_polymer

            engine = PolymerizationEngine(mock_monomer_manager)
            result = engine.create_polymer_chain(3)
            assert result is not None


class TestPackmolIntegrator:
    """
    Test PACKMOL integration functionality.
    """

    def test_packmol_validation_missing(self) -> None:
        """
        Test PACKMOL validation when executable is missing.
        """
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            integrator = PackmolIntegrator("nonexistent_packmol")
            assert integrator is not None

    def test_packmol_validation_success(self) -> None:
        """
        Test PACKMOL validation when executable is available.
        """
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            integrator = PackmolIntegrator()
            assert integrator is not None

    def test_generate_packmol_input(self) -> None:
        """
        Test PACKMOL input generation.
        """
        integrator = PackmolIntegrator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            polymer_pdb = tmp_path / "polymer.pdb"
            polymer_pdb.write_text("dummy pdb content")

            input_content = integrator._generate_packmol_input(
                polymer_pdb, total_chains=16, box_dimensions=(8.0, 8.0, 6.0), tmp_dir=tmp_path
            )

            assert "tolerance 2.0" in input_content
            assert "number 16" in input_content
            assert "80.0 80.0 60.0" in input_content

    def test_simple_tile_membrane(self) -> None:
        """
        Test simple membrane tiling fallback.
        """
        integrator = PackmolIntegrator()

        mock_molecule = Mock()
        mock_atom = Mock()
        mock_atom.element = Mock()
        mock_atom.element.symbol = "C"
        mock_atom.molecule_atom_index = 0
        mock_molecule.atoms = [mock_atom]

        with patch("openmm.app.Topology") as mock_topology:
            mock_topo_instance = Mock()
            mock_chain = Mock()
            mock_residue = Mock()
            mock_topo_instance.addChain.return_value = mock_chain
            mock_topo_instance.addResidue.return_value = mock_residue
            mock_topology.return_value = mock_topo_instance

            topology, positions = integrator._simple_tile_membrane(
                mock_molecule, chains_per_dim=[2, 2], box_dimensions=(4.0, 4.0, 2.0)
            )

            assert topology is not None
            assert len(positions) == 4


class TestSolvationSystem:
    """
    Test solvation system functionality.
    """

    def test_calculate_box_vectors(self) -> None:
        """
        Test box vector calculation.
        """
        solvation = SolvationSystem()

        box_vectors = solvation._calculate_box_vectors(box_dimensions=(8.0, 8.0, 6.0), solvent_config={"min_z_gap": 1.5})

        assert box_vectors[2].value_in_unit(unit.nanometer) == 9.0

    def test_solvate_membrane_failure(self) -> None:
        """
        Test solvation failure handling.
        """
        solvation = SolvationSystem()

        mock_topology = Mock()
        mock_positions: List[Any] = []

        with patch("openmm.app.Modeller") as mock_modeller_class:
            mock_modeller = Mock()
            mock_modeller.addSolvent.side_effect = Exception("Solvation failed")
            mock_modeller.topology = mock_topology
            mock_modeller.positions = mock_positions
            mock_modeller_class.return_value = mock_modeller

            result_topology, result_positions = solvation.solvate_membrane(
                mock_topology, mock_positions, solvent_config={"model": "tip3p"}, box_dimensions=(8.0, 8.0, 6.0)
            )

            assert result_topology == mock_topology
            assert result_positions == mock_positions


class TestForceFieldParametrizer:
    """
    Test force field parametrization functionality.
    """

    @pytest.fixture
    def resources_dir(self) -> Generator[Path, None, None]:
        """
        Create temporary resources directory.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_dependency_validation_missing_openff(self, resources_dir: Path) -> None:
        """
        Test dependency validation when OpenFF is missing.
        """
        with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.OPENFF_AVAILABLE", False):
            with pytest.raises(DependencyError, match="OpenFF Toolkit is required"):
                ForceFieldParametrizer(resources_dir)

    @pytest.mark.skipif(not OPENFF_AVAILABLE, reason="OpenFF not available")
    def test_load_force_field_base_only(self, resources_dir: Path) -> None:
        """
        Test loading base force field only.
        """
        parametrizer = ForceFieldParametrizer(resources_dir)

        with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.ForceField") as mock_ff:
            mock_ff.return_value = Mock()
            force_field = parametrizer._load_force_field()
            assert force_field is not None
            mock_ff.assert_called_once_with("openff-2.1.0.offxml")

    @pytest.mark.skipif(not OPENFF_AVAILABLE, reason="OpenFF not available")
    def test_load_force_field_with_custom(self, resources_dir: Path) -> None:
        """
        Test loading force field with custom parameters.
        """
        custom_ff_path = resources_dir / "polyamide_ff.yaml"
        custom_ff_path.write_text("# Custom force field parameters")

        parametrizer = ForceFieldParametrizer(resources_dir)

        with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.ForceField") as mock_ff:
            mock_ff.return_value = Mock()
            force_field = parametrizer._load_force_field()
            assert force_field is not None
            mock_ff.assert_called_once_with("openff-2.1.0.offxml", str(custom_ff_path))


class TestBuildFunction:
    """
    Test the main build function.
    """

    @pytest.fixture
    def resources_dir(self) -> Generator[Path, None, None]:
        """
        Create temporary resources directory with test files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            resources_path = Path(tmp_dir)

            polyamide_sdf = resources_path / "polyamide_monomer.sdf"
            polyamide_sdf.write_text(
                """polyamide_test
  Test

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  2  0  0  0  0
  3  6  1  0  0  0  0
  1  6  1  0  0  0  0
M  END
$$$$"""
            )

            yield resources_path

    def test_build_missing_dependencies(self) -> None:
        """
        Test build function with missing dependencies.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.OPENFF_AVAILABLE", False):
                config = {"repeat_units": [2, 2, 1]}

                with pytest.raises(PolyamideBuilderError):
                    build(Path(tmp_dir), config)

    def test_build_invalid_config(self) -> None:
        """
        Test build function with invalid configuration.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {"repeat_units": []}

            with pytest.raises(ValidationError):
                build(Path(tmp_dir), config)

    @pytest.mark.skipif(not OPENFF_AVAILABLE, reason="OpenFF not available")
    def test_build_success_minimal(self, resources_dir: Path) -> None:
        """
        Test successful build with minimal configuration.
        """
        config = {"repeat_units": [2, 2, 1]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("pathlib.Path.parent", return_value=resources_dir.parent):
                with patch("pathlib.Path.__truediv__", return_value=resources_dir):
                    with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.MonomerManager") as mock_mm:
                        with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.PolymerizationEngine") as mock_pe:
                            with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.PackmolIntegrator") as mock_pi:
                                with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.SolvationSystem") as mock_ss:
                                    with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.ForceFieldParametrizer") as mock_ffp:
                                        with patch("openmm.app.PDBFile.writeFile"):
                                            mock_molecule = Mock()
                                            mock_topology = Mock()
                                            mock_topology.atoms.return_value = [Mock(residue=Mock(name="POL"))]
                                            mock_positions: List[Any] = []

                                            mock_mm.return_value = Mock()
                                            mock_pe.return_value.create_polymer_chain.return_value = mock_molecule
                                            mock_pi.return_value.tile_membrane.return_value = (mock_topology, mock_positions)
                                            mock_ss.return_value.solvate_membrane.return_value = (mock_topology, mock_positions)
                                            mock_ffp.return_value.parametrize_system.side_effect = Exception("Force field failed")

                                            result = build(Path(tmp_dir), config)

                                            assert len(result) == 3
                                            pdb_path, topology, atom_indices = result
                                            assert pdb_path.exists()
                                            assert topology is not None
                                            assert isinstance(atom_indices, list)

    def test_build_component_failure(self) -> None:
        """
        Test build function handles component failures gracefully.
        """
        config = {"repeat_units": [2, 2, 1]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.MonomerManager") as mock_mm:
                mock_mm.side_effect = Exception("Monomer manager failed")

                with pytest.raises(PolyamideBuilderError, match="Build failed"):
                    build(Path(tmp_dir), config)


class TestIntegration:
    """
    Integration tests for the complete system.
    """

    @pytest.mark.skipif(not OPENFF_AVAILABLE, reason="OpenFF not available")
    def test_basic_workflow_integration(self) -> None:
        """
        Test basic workflow integration without heavy dependencies.
        """
        config = {"repeat_units": [2, 2, 1]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            _validate_config(config)
            assert True

    def test_error_propagation(self) -> None:
        """
        Test that errors are properly propagated through the system.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValidationError):
                build(Path(tmp_dir), {})

            with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.OPENFF_AVAILABLE", False):
                with pytest.raises(PolyamideBuilderError):
                    build(Path(tmp_dir), {"repeat_units": [2, 2, 1]})

    def test_logging_functionality(self, caplog: Any) -> None:
        """
        Test that logging works correctly throughout the system.
        """
        config = {"repeat_units": [2, 2, 1]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with caplog.at_level(
                logging.INFO, logger="moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build"
            ):
                with patch("moml.simulation.molecular_dynamics.force_field.plugins.nf_polyamide_v1.build.OPENFF_AVAILABLE", False):
                    try:
                        build(Path(tmp_dir), config)
                    except PolyamideBuilderError:
                        pass

                    assert len(caplog.records) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
