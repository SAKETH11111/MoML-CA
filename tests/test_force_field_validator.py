"""
tests/test_force_field_validator.py

Unit tests for the ForceFieldValidator class in
moml.simulation.molecular_dynamics.force_field.validator.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch
from typing import Generator

import pytest

from moml.simulation.molecular_dynamics.force_field.validator import ForceFieldValidator


class TestForceFieldValidator:
    """
    Test suite for ForceFieldValidator class.
    """

    @pytest.fixture
    def sample_forcefield_xml(self) -> Path:
        """
        Create a sample force field XML file for testing.
        """
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <ForceField>
            <AtomTypes>
                <Type name="H" class="H" element="H" mass="1.008"/>
                <Type name="O" class="O" element="O" mass="15.999"/>
            </AtomTypes>
            <HarmonicBondForce>
                <Bond class1="O" class2="H" length="0.09572" k="462750.4"/>
            </HarmonicBondForce>
            <HarmonicAngleForce>
                <Angle class1="H" class2="O" class3="H" angle="1.824218" k="836.8"/>
            </HarmonicAngleForce>
            <PeriodicTorsionForce>
                <Proper class1="H" class2="O" class3="O" class4="H" k="10.0" periodicity="1" phase="0"/>
            </PeriodicTorsionForce>
            <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
                <Atom type="H" charge="0.417" sigma="0.0" epsilon="0.0"/>
                <Atom type="O" charge="-0.834" sigma="0.315061" epsilon="0.635968"/>
            </NonbondedForce>
        </ForceField>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            return Path(f.name)

    @pytest.fixture
    def incomplete_forcefield_xml(self) -> Path:
        """
        Create an incomplete force field XML file for testing.
        """
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
        <ForceField>
            <AtomTypes>
                <Type name="H" class="H" element="H" mass="1.008"/>
            </AtomTypes>
            <HarmonicBondForce>
                <Bond class1="O" class2="H" length="0.09572" k="462750.4"/>
            </HarmonicBondForce>
        </ForceField>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            return Path(f.name)

    def test_validator_initialization(self, sample_forcefield_xml: Path) -> None:
        """
        Test that validator initializes correctly with valid XML.
        """
        validator = ForceFieldValidator(sample_forcefield_xml)
        assert validator.xml_path == sample_forcefield_xml
        assert validator.root.tag == 'ForceField'
        sample_forcefield_xml.unlink()

    def test_validate_completeness_success(self, sample_forcefield_xml: Path) -> None:
        """
        Test completeness validation with complete force field.
        """
        validator = ForceFieldValidator(sample_forcefield_xml)
        errors = validator.validate_completeness()
        assert len(errors) == 0
        sample_forcefield_xml.unlink()

    def test_validate_completeness_missing_sections(self, incomplete_forcefield_xml: Path) -> None:
        """
        Test completeness validation with missing sections.
        """
        validator = ForceFieldValidator(incomplete_forcefield_xml)
        errors = validator.validate_completeness()
        assert len(errors) == 3
        assert any("HarmonicAngleForce" in error for error in errors)
        assert any("PeriodicTorsionForce" in error for error in errors)
        assert any("NonbondedForce" in error for error in errors)
        incomplete_forcefield_xml.unlink()

    def test_smoke_test_signature_and_parameters(self, sample_forcefield_xml: Path) -> None:
        """
        Test that smoke_test accepts new parameters correctly.
        """
        validator = ForceFieldValidator(sample_forcefield_xml)
        try:
            validator.smoke_test.__func__.__code__.co_varnames
            expected_params = {'self', 'steps', 'pdb_path', 'deep_check', 'water_box_size_nm', 'n_waters'}
            actual_params = set(validator.smoke_test.__func__.__code__.co_varnames[:6])
            assert expected_params.issubset(actual_params), f"Missing parameters: {expected_params - actual_params}"
        except Exception as e:
            pytest.fail(f"Method signature test failed: {e}")
        sample_forcefield_xml.unlink()

    def test_build_minimal_water_box(self, sample_forcefield_xml: Path) -> None:
        """
        Test the _build_minimal_water_box helper method.
        """
        try:
            import openmm.app as app
            import openmm
        except ImportError:
            pytest.skip("OpenMM not available")
        validator = ForceFieldValidator(sample_forcefield_xml)
        topology, positions = validator._build_minimal_water_box(2.0, 2)
        assert len(positions) == 6
        atoms = list(topology.atoms())
        assert len(atoms) == 6
        oxygen_count = sum(1 for atom in atoms if atom.element.symbol == 'O')
        hydrogen_count = sum(1 for atom in atoms if atom.element.symbol == 'H')
        assert oxygen_count == 2
        assert hydrogen_count == 4
        bonds = list(topology.bonds())
        assert len(bonds) == 4
        sample_forcefield_xml.unlink()

    def test_build_minimal_water_box_edge_cases(self, sample_forcefield_xml: Path) -> None:
        """
        Test _build_minimal_water_box with edge cases.
        """
        try:
            import openmm.app as app
            import openmm
        except ImportError:
            pytest.skip("OpenMM not available")
        validator = ForceFieldValidator(sample_forcefield_xml)
        topology, positions = validator._build_minimal_water_box(1.0, 1)
        assert len(positions) == 3
        topology, positions = validator._build_minimal_water_box(1.0, 0)
        assert len(positions) == 0
        sample_forcefield_xml.unlink()

    def test_validate_ranges(self, sample_forcefield_xml: Path) -> None:
        """
        Test parameter range validation.
        """
        validator = ForceFieldValidator(sample_forcefield_xml)
        errors = validator.validate_ranges()
        assert len(errors) == 0
        sample_forcefield_xml.unlink()

    def test_validate_all_checks(self, sample_forcefield_xml: Path) -> None:
        """
        Test the main validate method that runs all checks.
        """
        with patch.object(ForceFieldValidator, 'smoke_test', return_value=[]):
            validator = ForceFieldValidator(sample_forcefield_xml)
            results = validator.validate()
            assert 'completeness' in results
            assert 'ranges' in results
            assert 'smoke_test' in results
            assert len(results['completeness']) == 0
            assert len(results['ranges']) == 0
            assert len(results['smoke_test']) == 0
        sample_forcefield_xml.unlink()