"""
tests/test_orca_parser.py

Unit tests for the ORCA parser and calculation management functions
in moml.simulation.qm.parser.orca_parser.
"""

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import pytest
from rdkit import Chem

from moml.simulation.qm.parser.orca_parser import (
    batch_process_molecules,
    create_orca_input,
    extract_partial_charges_from_orca,
    parse_orca_output,
    process_molecule,
    run_orca_calculation,
    smiles_to_3d_structure,
)

# --- Test Data for ORCA Output Parsing ---
DUMMY_ORCA_OUTPUT_COMPLETE = """
Some initial lines...
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
 C          0.000000    0.000000    0.000000
 H          1.000000    0.000000    0.000000
 H          0.000000    1.000000    0.000000

More lines...
MULLIKEN ATOMIC CHARGES
-----------------------
   0 C   :    -0.500000
   1 H   :     0.250000
   2 H   :     0.250000

LOEWDIN ATOMIC CHARGES
----------------------
   0 C   :    -0.400000
   1 H   :     0.200000
   2 H   :     0.200000
   
DIPOLE MOMENT
-------------
               X           Y           Z         Total
Electronic  0.100000    0.200000    0.300000    0.374166
Nuclear     ...
Total       0.100000    0.200000    0.300000    0.374166 a.u.

ORBITAL ENERGIES
-----------------
...
   20  -0.500000000000   1.000000000000  (HOMO)
   21  -0.100000000000   0.000000000000  (LUMO)
...
HOMO-LUMO gap:         0.400000 Eh  =     10.8844 eV

****ORCA TERMINATED NORMALLY****
"""

DUMMY_ORCA_OUTPUT_ERROR = """
Some initial lines...
CARTESIAN COORDINATES (ANGSTROEM)
 C          0.000000    0.000000    0.000000
Error in SCF calculation!
****ORCA TERMINATED ABNORMALLY****
"""

DUMMY_ORCA_OUTPUT_INCOMPLETE = """
Some initial lines...
MULLIKEN ATOMIC CHARGES
   0 C   :    -0.500000
(file ends abruptly)
"""

DUMMY_ORCA_OUTPUT_NO_LOEWDIN_NO_DIRECT_GAP = """
CARTESIAN COORDINATES (ANGSTROEM)
 C          0.000000    0.000000    0.000000
MULLIKEN ATOMIC CHARGES
   0 C   :    -0.500000
ORBITAL ENERGIES
   0  -10.000000000000   2.000000000000  (HOMO)
   1   -0.200000000000   2.000000000000  (LUMO)
****ORCA TERMINATED NORMALLY****
"""


@pytest.fixture
def temp_orca_output_file() -> Generator[str, None, None]:
    """
    Creates a temporary file and returns its path.

    Yields:
        str: Path to the temporary file.
    """
    fd, path = tempfile.mkstemp(text=True)
    os.close(fd)
    yield path
    os.remove(path)


class TestParseOrcaOutput:
    """
    Test suite for the parse_orca_output function.
    """

    def test_parse_complete_output(self, temp_orca_output_file: str) -> None:
        """
        Test parsing a complete ORCA output file.
        """
        with open(temp_orca_output_file, "w") as f:
            f.write(DUMMY_ORCA_OUTPUT_COMPLETE)

        results = parse_orca_output(temp_orca_output_file)

        assert results["status"] == "completed"
        assert len(results["mulliken_charges"]) == 3
        assert results["mulliken_charges"] == pytest.approx([-0.5, 0.25, 0.25])
        assert len(results["loewdin_charges"]) == 3
        assert results["loewdin_charges"] == pytest.approx([-0.4, 0.20, 0.20])
        assert results["dipole_moment"] is not None
        assert results["dipole_moment"] == pytest.approx([0.1, 0.2, 0.3, 0.374166])
        assert results["homo_lumo_gap"] is not None
        assert results["homo_lumo_gap"] == pytest.approx(10.8844)
        assert len(results["optimized_geometry"]) == 3
        assert results["optimized_geometry"][0]["symbol"] == "C"
        assert results["optimized_geometry"][1]["coordinates"] == pytest.approx([1.0, 0.0, 0.0])

    def test_parse_error_output(self, temp_orca_output_file: str) -> None:
        """
        Test parsing an ORCA output file with an error.
        """
        with open(temp_orca_output_file, "w") as f:
            f.write(DUMMY_ORCA_OUTPUT_ERROR)
        results = parse_orca_output(temp_orca_output_file)
        assert results["status"] == "error"

    def test_parse_incomplete_output(self, temp_orca_output_file: str) -> None:
        """
        Test parsing an incomplete ORCA output file.
        """
        with open(temp_orca_output_file, "w") as f:
            f.write(DUMMY_ORCA_OUTPUT_INCOMPLETE)
        results = parse_orca_output(temp_orca_output_file)
        assert results["status"] == "incomplete"
        assert results["mulliken_charges"] == pytest.approx([-0.5])

    def test_parse_no_loewdin_no_direct_gap(self, temp_orca_output_file: str) -> None:
        """
        Test parsing ORCA output without Loewdin charges or direct HOMO-LUMO gap.
        """
        with open(temp_orca_output_file, "w") as f:
            f.write(DUMMY_ORCA_OUTPUT_NO_LOEWDIN_NO_DIRECT_GAP)
        results = parse_orca_output(temp_orca_output_file)
        assert results["status"] == "completed"
        assert len(results["loewdin_charges"]) == 0
        assert results["dipole_moment"] is None
        assert results["homo_lumo_gap"] is not None
        assert results["homo_lumo_gap"] == pytest.approx(9.8 * 27.211)

    def test_file_not_found(self) -> None:
        """
        Test FileNotFoundError when parsing a non-existent file.
        """
        with pytest.raises(FileNotFoundError):
            parse_orca_output("non_existent_file.out")


class TestExtractPartialCharges:
    """
    Test suite for the extract_partial_charges_from_orca function.
    """

    @patch("moml.simulation.qm.parser.orca_parser.parse_orca_output")
    def test_extract_mulliken(self, mock_parse: MagicMock) -> None:
        """
        Test extracting Mulliken partial charges.
        """
        mock_parse.return_value = {"mulliken_charges": [1.0, -1.0], "loewdin_charges": [0.5, -0.5]}
        charges = extract_partial_charges_from_orca("dummy.out", charge_type="mulliken")
        assert charges == [1.0, -1.0]

    @patch("moml.simulation.qm.parser.orca_parser.parse_orca_output")
    def test_extract_loewdin(self, mock_parse: MagicMock) -> None:
        """
        Test extracting Loewdin partial charges.
        """
        mock_parse.return_value = {"mulliken_charges": [1.0, -1.0], "loewdin_charges": [0.5, -0.5]}
        charges = extract_partial_charges_from_orca("dummy.out", charge_type="loewdin")
        assert charges == [0.5, -0.5]

    @patch("moml.simulation.qm.parser.orca_parser.parse_orca_output")
    def test_extract_unknown_type(self, mock_parse: MagicMock, caplog: Any) -> None:
        """
        Test extracting charges with an unknown charge type (should default to Mulliken).
        """
        mock_parse.return_value = {"mulliken_charges": [1.0, -1.0], "loewdin_charges": [0.5, -0.5]}
        charges = extract_partial_charges_from_orca("dummy.out", charge_type="unknown")
        assert charges == [1.0, -1.0]
        assert "Unknown charge type 'unknown'" in caplog.text


class TestSmilesTo3DStructure:
    """
    Test suite for the smiles_to_3d_structure function.
    """

    def test_valid_smiles(self) -> None:
        """
        Test converting a valid SMILES string to a 3D RDKit molecule.
        """
        mol = smiles_to_3d_structure("CCO", "ethanol_test")
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumConformers() > 0

    def test_invalid_smiles(self, caplog: Any) -> None:
        """
        Test converting an invalid SMILES string (should return None and log error).
        """
        mol = smiles_to_3d_structure("InvalidSMILES", "invalid_test")
        assert mol is None
        assert "Failed to parse SMILES: InvalidSMILES" in caplog.text

    @patch("rdkit.Chem.AllChem.EmbedMolecule", return_value=-1)
    def test_embed_failure(self, mock_embed: MagicMock, caplog: Any) -> None:
        """
        Test handling of EmbedMolecule failure (should return None and log error).
        """
        mol = smiles_to_3d_structure("C", "methane_embed_fail")
        assert mol is None
        assert "Coordinate generation failed for methane_embed_fail" in caplog.text


class TestCreateOrcaInput:
    """
    Test suite for the create_orca_input function.
    """

    @pytest.fixture
    def temp_dir_for_orca(self) -> Generator[str, None, None]:
        """
        Creates a temporary directory for ORCA input/output files.
        """
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    def test_create_input_file(self, temp_dir_for_orca: str) -> None:
        """
        Test creating a basic ORCA input file.
        """
        mol = smiles_to_3d_structure("C", "methane_test")
        assert mol is not None

        success, inp_path = create_orca_input(
            mol, "methane_test", temp_dir_for_orca, functional="wB97X-D", basis_set="def2-SVP", num_procs=2, memory=2000
        )
        assert success
        assert os.path.exists(inp_path)
        assert os.path.exists(os.path.join(temp_dir_for_orca, "methane_test.mol"))

        with open(inp_path, "r") as f:
            content = f.read()
            assert "! wB97X-D3 def2-SVP OPT" in content
            assert "%pal" in content
            assert "nprocs 2" in content
            assert "%maxcore 2000" in content
            assert "* xyz 0 1" in content
            assert "  C " in content
            assert "  H " in content

    def test_create_input_b3lyp(self, temp_dir_for_orca: str) -> None:
        """
        Test creating ORCA input with B3LYP functional.
        """
        mol = smiles_to_3d_structure("C", "methane_b3lyp")
        assert mol is not None
        success, inp_path = create_orca_input(mol, "methane_b3lyp", temp_dir_for_orca, functional="B3LYP")
        assert success
        with open(inp_path, "r") as f:
            content = f.read()
            assert "! B3LYP D3BJ 6-31G* OPT" in content

    def test_create_input_single_proc(self, temp_dir_for_orca: str) -> None:
        """
        Test creating ORCA input for single processor.
        """
        mol = smiles_to_3d_structure("C", "methane_sp")
        assert mol is not None
        success, inp_path = create_orca_input(mol, "methane_sp", temp_dir_for_orca, num_procs=1)
        assert success
        with open(inp_path, "r") as f:
            content = f.read()
            assert "%pal" not in content


@patch("subprocess.run")
@patch("os.path.exists")
class TestRunOrcaCalculation:
    """
    Test suite for the run_orca_calculation function.
    """

    def test_run_successful(self, mock_exists: MagicMock, mock_run: MagicMock) -> None:
        """
        Test a successful ORCA calculation run.
        """
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".inp", delete=False) as tmp_inp_f:
            input_file_path = tmp_inp_f.name
            tmp_inp_f.write(b"dummy orca input content")

        output_file_expected = input_file_path.replace(".inp", ".out")
        success = False
        try:
            mock_exists.side_effect = lambda p: p == "orca" or p == input_file_path or p == output_file_expected
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK", stderr="")

            if mock_run.return_value.returncode == 0:
                with open(output_file_expected, "w") as f_out:
                    f_out.write("dummy orca output for test_run_successful")

            success, out_path = run_orca_calculation(input_file_path, orca_path="orca")

            assert success
            assert out_path == output_file_expected
            mock_run.assert_called_once_with(
                ["orca", input_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(input_file_path),
                check=False,
            )
        finally:
            if os.path.exists(input_file_path):
                os.remove(input_file_path)
            if success and os.path.exists(output_file_expected):
                os.remove(output_file_expected)

    def test_run_orca_fail_returncode(self, mock_exists: MagicMock, mock_run: MagicMock) -> None:
        """
        Test ORCA calculation failure due to non-zero return code.
        """
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".inp", delete=False) as tmp_inp_f:
            input_file_path = tmp_inp_f.name
            tmp_inp_f.write(b"dummy orca input content")

        output_file_expected = input_file_path.replace(".inp", ".out")

        try:
            if os.path.exists(output_file_expected):
                try:
                    os.remove(output_file_expected)
                except FileNotFoundError:
                    pass

            mock_exists.side_effect = lambda p: p == "orca" or p == input_file_path
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="ORCA ERROR")
            success, out_path = run_orca_calculation(input_file_path, orca_path="orca")
            assert not success
            assert out_path == output_file_expected

        finally:
            try:
                if os.path.exists(input_file_path):
                    os.remove(input_file_path)
            except FileNotFoundError:
                pass
            try:
                if os.path.exists(output_file_expected):
                    os.remove(output_file_expected)
            except FileNotFoundError:
                pass

    def test_run_output_not_created(self, mock_exists: MagicMock, mock_run: MagicMock) -> None:
        """
        Test ORCA calculation where output file is not created despite successful run.
        """
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".inp", delete=False) as tmp_inp_f:
            input_file_path = tmp_inp_f.name
            tmp_inp_f.write(b"dummy orca input content")

        output_file_path = input_file_path.replace(".inp", ".out")

        try:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
        except Exception:
            pass

        def mock_exists_logic(path_to_check: str) -> bool:
            if path_to_check == input_file_path:
                return True
            if path_to_check == output_file_path:
                return False
            return False

        mock_exists.side_effect = mock_exists_logic
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK", stderr="")
        success, out_path = run_orca_calculation(input_file_path, orca_path="orca")
        assert not success
        assert out_path == output_file_path


@patch("moml.simulation.qm.parser.orca_parser.smiles_to_3d_structure")
@patch("moml.simulation.qm.parser.orca_parser.create_orca_input")
@patch("moml.simulation.qm.parser.orca_parser.run_orca_calculation")
@patch("moml.simulation.qm.parser.orca_parser.parse_orca_output")
class TestProcessMolecule:
    """
    Test suite for the process_molecule function.
    """

    def test_process_molecule_success(
        self,
        mock_parse: MagicMock,
        mock_run: MagicMock,
        mock_create_inp: MagicMock,
        mock_smiles_3d: MagicMock,
        temp_orca_output_file: str,
    ) -> None:
        """
        Test successful processing of a single molecule.
        """
        mol_dir = os.path.dirname(temp_orca_output_file)
        mol_id = "test_mol"

        mock_smiles_3d.return_value = Chem.MolFromSmiles("C")
        mock_create_inp.return_value = (True, os.path.join(mol_dir, f"{mol_id}.inp"))
        mock_run.return_value = (True, os.path.join(mol_dir, f"{mol_id}.out"))
        mock_parse.return_value = {"status": "completed", "data_key": "value"}

        results_path = os.path.join(mol_dir, mol_id, f"{mol_id}_results.json")
        with patch("builtins.open", mock_open()) as mock_file_write:
            results = process_molecule("C", mol_id, mol_dir, "B3LYP", "def2-SVP", 1, 1000, "orca")

        assert results["status"] == "completed"
        assert results["data"] == {"status": "completed", "data_key": "value"}
        assert "error" not in results
        mock_file_write.assert_called_once_with(results_path, "w")

    def test_process_molecule_smiles_fail(
        self,
        mock_parse: MagicMock,
        mock_run: MagicMock,
        mock_create_inp: MagicMock,
        mock_smiles_3d: MagicMock,
        temp_orca_output_file: str,
    ) -> None:
        """
        Test processing a molecule fails due to SMILES conversion.
        """
        mol_dir = os.path.dirname(temp_orca_output_file)
        mock_smiles_3d.return_value = None

        results = process_molecule("C", "fail_mol", mol_dir, "B3LYP", "def2-SVP", 1, 1000, "orca")
        assert results["status"] == "failed"
        assert results["error"] == "Failed to create 3D structure"
        mock_create_inp.assert_not_called()


@pytest.fixture
def sample_molecules_df() -> pd.DataFrame:
    """
    Provides a sample DataFrame for batch processing tests.
    """
    data = {"id": ["mol1", "mol2", "mol3"], "smiles": ["C", "CC", "CCC"]}
    return pd.DataFrame(data)


@pytest.fixture
def temp_batch_output_dir() -> Generator[str, None, None]:
    """
    Creates a temporary directory for batch output files.
    """
    dir_path = tempfile.mkdtemp(prefix="batch_orca_")
    yield dir_path
    shutil.rmtree(dir_path)


@patch("moml.simulation.qm.parser.orca_parser.process_molecule")
@patch("concurrent.futures.ProcessPoolExecutor")
class TestBatchProcessMolecules:
    """
    Test suite for the batch_process_molecules function.
    """

    @pytest.mark.skip(reason="Test hangs or is too slow, to be investigated later")
    def test_batch_process_all_success(
        self,
        mock_executor_cls: MagicMock,
        mock_process_mol: MagicMock,
        sample_molecules_df: pd.DataFrame,
        temp_batch_output_dir: str,
    ) -> None:
        """
        Test batch_process_molecules with all individual molecule processes succeeding.
        """
        mock_process_mol.side_effect = [
            {"id": "mol1", "smiles": "C", "status": "completed", "data": {"prop": 1}, "error": None},
            {"id": "mol2", "smiles": "CC", "status": "completed", "data": {"prop": 2}, "error": None},
            {"id": "mol3", "smiles": "CCC", "status": "completed", "data": {"prop": 3}, "error": None},
        ]

        mock_executor_instance = MagicMock(spec=ProcessPoolExecutor)
        mock_executor_cls.return_value.__enter__.return_value = mock_executor_instance

        future1, future2, future3 = MagicMock(), MagicMock(), MagicMock()
        future1.result.return_value = {
            "id": "mol1",
            "smiles": "C",
            "status": "completed",
            "data": {"prop": 1},
            "error": None,
        }
        future2.result.return_value = {
            "id": "mol2",
            "smiles": "CC",
            "status": "failed",
            "data": None,
            "error": "ORCA Error",
        }
        future3.result.return_value = {
            "id": "mol3",
            "smiles": "CCC",
            "status": "completed",
            "data": {"prop": 3},
            "error": None,
        }

        mock_executor_instance.submit.side_effect = [future1, future2, future3]

        results = batch_process_molecules(
            molecules_df=sample_molecules_df,
            output_dir=temp_batch_output_dir,
            functional="B3LYP",
            basis_set="def2-SVP",
            num_procs=2,
            memory=2000,
            orca_path="orca_mock",
            max_workers=2,
            smiles_col="smiles",
            id_col="id",
        )

        assert len(results) == 3
        assert results[0]["status"] == "completed"
        assert results[1]["data"]["prop"] == 2

        expected_calls = [
            call("C", "mol1", temp_batch_output_dir, "B3LYP", "def2-SVP", 2, 2000, "orca_mock"),
            call("CC", "mol2", temp_batch_output_dir, "B3LYP", "def2-SVP", 2, 2000, "orca_mock"),
            call("CCC", "mol3", temp_batch_output_dir, "B3LYP", "def2-SVP", 2, 2000, "orca_mock"),
        ]
        mock_process_mol.assert_has_calls(expected_calls, any_order=False)
        assert mock_process_mol.call_count == 3

        summary_file_path = os.path.join(temp_batch_output_dir, "orca_batch_summary.json")
        assert os.path.exists(summary_file_path)
        with open(summary_file_path, "r") as f:
            summary_data = json.load(f)
        assert len(summary_data) == 3
        assert summary_data[0]["id"] == "mol1"

    @pytest.mark.skip(reason="Test hangs or is too slow, to be investigated later")
    def test_batch_process_with_failures(
        self,
        mock_executor_cls: MagicMock,
        mock_process_mol: MagicMock,
        sample_molecules_df: pd.DataFrame,
        temp_batch_output_dir: str,
    ) -> None:
        """
        Test batch_process_molecules with some individual molecule processes failing.
        """
        mock_process_mol.side_effect = [
            {"id": "mol1", "smiles": "C", "status": "completed", "data": {"prop": 1}, "error": None},
            {"id": "mol2", "smiles": "CC", "status": "failed", "data": None, "error": "ORCA Error"},
            {"id": "mol3", "smiles": "CCC", "status": "completed", "data": {"prop": 3}, "error": None},
        ]

        mock_executor_instance = MagicMock(spec=ProcessPoolExecutor)
        mock_executor_cls.return_value.__enter__.return_value = mock_executor_instance

        future1, future2, future3 = MagicMock(), MagicMock(), MagicMock()
        future1.result.return_value = {
            "id": "mol1",
            "smiles": "C",
            "status": "completed",
            "data": {"prop": 1},
            "error": None,
        }
        future2.result.return_value = {
            "id": "mol2",
            "smiles": "CC",
            "status": "failed",
            "data": None,
            "error": "ORCA Error",
        }
        future3.result.return_value = {
            "id": "mol3",
            "smiles": "CCC",
            "status": "completed",
            "data": {"prop": 3},
            "error": None,
        }

        mock_executor_instance.submit.side_effect = [future1, future2, future3]

        results = batch_process_molecules(
            molecules_df=sample_molecules_df,
            output_dir=temp_batch_output_dir,
            functional="B3LYP",
            basis_set="def2-SVP",
            num_procs=1,
            memory=1000,
            orca_path="orca_mock",
            smiles_col="smiles",
            id_col="id",
        )

        assert len(results) == 3
        assert results[0]["status"] == "completed"
        assert results[1]["status"] == "failed"
        assert results[1]["error"] == "ORCA Error"
        assert results[2]["status"] == "completed"

        summary_file_path = os.path.join(temp_batch_output_dir, "orca_batch_summary.json")
        assert os.path.exists(summary_file_path)
        with open(summary_file_path, "r") as f:
            summary_data = json.load(f)
        assert len(summary_data) == 3
        assert summary_data[1]["id"] == "mol2"
        assert summary_data[1]["status"] == "failed"

    @pytest.mark.skip(reason="Test hangs or is too slow, to be investigated later")
    def test_batch_process_future_exception(
        self,
        mock_executor_cls: MagicMock,
        mock_process_mol: MagicMock,
        sample_molecules_df: pd.DataFrame,
        temp_batch_output_dir: str,
        caplog: Any,
    ) -> None:
        """
        Test batch_process_molecules when a future raises an exception.
        """
        mock_executor_instance = MagicMock(spec=ProcessPoolExecutor)
        mock_executor_cls.return_value.__enter__.return_value = mock_executor_instance

        future1, future2, future3 = MagicMock(), MagicMock(), MagicMock()
        future1.result.return_value = {
            "id": "mol1",
            "smiles": "C",
            "status": "completed",
            "data": {"prop": 1},
            "error": None,
        }
        future2.result.side_effect = Exception("Simulated future error")
        future3.result.return_value = {
            "id": "mol3",
            "smiles": "CCC",
            "status": "completed",
            "data": {"prop": 3},
            "error": None,
        }

        mock_executor_instance.submit.side_effect = [future1, future2, future3]

        results = batch_process_molecules(
            molecules_df=sample_molecules_df,
            output_dir=temp_batch_output_dir,
            functional="B3LYP",
            basis_set="def2-SVP",
            num_procs=1,
            memory=1000,
            orca_path="orca_mock",
            smiles_col="smiles",
            id_col="id",
        )

        assert len(results) == 3
        assert results[0]["status"] == "completed"

        assert results[1]["id"] == "mol2"
        assert results[1]["status"] == "error"
        assert "Simulated future error" in results[1]["error_message"]
        assert "Error processing molecule mol2" in caplog.text

        assert results[2]["status"] == "completed"

        summary_file_path = os.path.join(temp_batch_output_dir, "orca_batch_summary.json")
        assert os.path.exists(summary_file_path)
        with open(summary_file_path, "r") as f:
            summary_data = json.load(f)
        assert len(summary_data) == 3
        assert summary_data[1]["status"] == "error"
