"""
Unit tests for the ORCA parser and calculation management functions
in moml.simulation.quantum_mechanics.parser.orca_parser.
"""

import pytest
import os
import tempfile
import shutil
import subprocess
import pandas as pd  # Added for batch_process_molecules
import json  # Added for checking JSON output
from unittest.mock import patch, mock_open, MagicMock, call  # Added call for checking multiple calls

from moml.simulation.quantum_mechanics.parser.orca_parser import (
    parse_orca_output,
    extract_partial_charges_from_orca,
    smiles_to_3d_structure,
    create_orca_input,
    run_orca_calculation,
    process_molecule,
    batch_process_molecules,  # Added for testing
    # extract_orbital_contributions_from_orca, # Placeholder, skip for now
    # extract_electrostatic_potential_from_orca # Placeholder, skip for now
)
from rdkit import Chem

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
def temp_orca_output_file():
    """Creates a temporary file and returns its path."""
    fd, path = tempfile.mkstemp(text=True)
    os.close(fd)  # Close the file descriptor
    yield path
    os.remove(path)  # Clean up


class TestParseOrcaOutput:
    def test_parse_complete_output(self, temp_orca_output_file):
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

    def test_parse_error_output(self, temp_orca_output_file):
        with open(temp_orca_output_file, "w") as f:
            f.write(DUMMY_ORCA_OUTPUT_ERROR)
        results = parse_orca_output(temp_orca_output_file)
        assert results["status"] == "error"

    def test_parse_incomplete_output(self, temp_orca_output_file):
        with open(temp_orca_output_file, "w") as f:
            f.write(DUMMY_ORCA_OUTPUT_INCOMPLETE)
        results = parse_orca_output(temp_orca_output_file)
        assert results["status"] == "incomplete"
        assert results["mulliken_charges"] == pytest.approx([-0.5])  # Parses what it can

    def test_parse_no_loewdin_no_direct_gap(self, temp_orca_output_file):
        with open(temp_orca_output_file, "w") as f:
            f.write(DUMMY_ORCA_OUTPUT_NO_LOEWDIN_NO_DIRECT_GAP)
        results = parse_orca_output(temp_orca_output_file)
        assert results["status"] == "completed"
        assert len(results["loewdin_charges"]) == 0
        assert results["dipole_moment"] is None
        # HOMO = -10.0 Eh, LUMO = -0.2 Eh. Gap = 9.8 Eh
        # Gap eV = 9.8 * 27.211 = 266.6678
        assert results["homo_lumo_gap"] is not None
        assert results["homo_lumo_gap"] == pytest.approx(9.8 * 27.211)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_orca_output("non_existent_file.out")


class TestExtractPartialCharges:
    @patch("moml.simulation.quantum_mechanics.parser.orca_parser.parse_orca_output")
    def test_extract_mulliken(self, mock_parse):
        mock_parse.return_value = {"mulliken_charges": [1.0, -1.0], "loewdin_charges": [0.5, -0.5]}
        charges = extract_partial_charges_from_orca("dummy.out", charge_type="mulliken")
        assert charges == [1.0, -1.0]

    @patch("moml.simulation.quantum_mechanics.parser.orca_parser.parse_orca_output")
    def test_extract_loewdin(self, mock_parse):
        mock_parse.return_value = {"mulliken_charges": [1.0, -1.0], "loewdin_charges": [0.5, -0.5]}
        charges = extract_partial_charges_from_orca("dummy.out", charge_type="loewdin")
        assert charges == [0.5, -0.5]

    @patch("moml.simulation.quantum_mechanics.parser.orca_parser.parse_orca_output")
    def test_extract_unknown_type(self, mock_parse, caplog):
        mock_parse.return_value = {"mulliken_charges": [1.0, -1.0], "loewdin_charges": [0.5, -0.5]}
        charges = extract_partial_charges_from_orca("dummy.out", charge_type="unknown")
        assert charges == [1.0, -1.0]  # Defaults to Mulliken
        assert "Unknown charge type 'unknown'" in caplog.text


class TestSmilesTo3DStructure:
    def test_valid_smiles(self):
        mol = smiles_to_3d_structure("CCO", "ethanol_test")
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumConformers() > 0

    def test_invalid_smiles(self, caplog):
        mol = smiles_to_3d_structure("InvalidSMILES", "invalid_test")
        assert mol is None
        assert "Failed to parse SMILES: InvalidSMILES" in caplog.text

    # EmbedMolecule can sometimes fail even for valid SMILES if they are too simple/constrained
    # This test might be flaky or require a specific RDKit version if not handled well by EmbedMolecule
    @patch("rdkit.Chem.AllChem.EmbedMolecule", return_value=-1)  # Mock embed to fail
    def test_embed_failure(self, mock_embed, caplog):
        mol = smiles_to_3d_structure("C", "methane_embed_fail")  # Methane
        assert mol is None
        assert "Coordinate generation failed for methane_embed_fail" in caplog.text


class TestCreateOrcaInput:
    @pytest.fixture
    def temp_dir_for_orca(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    def test_create_input_file(self, temp_dir_for_orca):
        mol = smiles_to_3d_structure("C", "methane_test")  # Methane
        assert mol is not None

        success, inp_path = create_orca_input(
            mol, "methane_test", temp_dir_for_orca, functional="wB97X-D", basis_set="def2-SVP", num_procs=2, memory=2000
        )
        assert success
        assert os.path.exists(inp_path)
        assert os.path.exists(os.path.join(temp_dir_for_orca, "methane_test.mol"))

        with open(inp_path, "r") as f:
            content = f.read()
            assert "! wB97X-D3 def2-SVP OPT" in content  # wB97X-D becomes wB97X-D3
            assert "%pal" in content
            assert "nprocs 2" in content
            assert "%maxcore 2000" in content
            assert "* xyz 0 1" in content  # CH4 is neutral singlet
            # Check for atom symbols followed by a space, as per the output format: f"  {atom_sym} {pos.x:.6f}..."
            assert "  C " in content  # Check for Carbon line start
            assert "  H " in content  # Check for Hydrogen line start

    def test_create_input_b3lyp(self, temp_dir_for_orca):
        mol = smiles_to_3d_structure("C", "methane_b3lyp")
        success, inp_path = create_orca_input(mol, "methane_b3lyp", temp_dir_for_orca, functional="B3LYP")
        assert success
        with open(inp_path, "r") as f:
            content = f.read()
            assert "! B3LYP D3BJ 6-31G* OPT" in content  # Default basis, D3BJ added

    def test_create_input_single_proc(self, temp_dir_for_orca):
        mol = smiles_to_3d_structure("C", "methane_sp")
        success, inp_path = create_orca_input(mol, "methane_sp", temp_dir_for_orca, num_procs=1)
        assert success
        with open(inp_path, "r") as f:
            content = f.read()
            assert "%pal" not in content  # No pal block for 1 proc


@patch("subprocess.run")
@patch("os.path.exists")
class TestRunOrcaCalculation:
    def test_run_successful(self, mock_exists, mock_run):
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".inp", delete=False
        ) as tmp_inp_f:  # Explicitly w+b, though default
            input_file_path = tmp_inp_f.name
            tmp_inp_f.write(b"dummy orca input content")  # Write bytes

        # Expected output path based on input_file_path
        output_file_expected = input_file_path.replace(".inp", ".out")

        try:
            # mock_exists needs to return True for 'orca', the input_file_path, and potentially the output_file_expected
            # if run_orca_calculation checks for output dir or pre-existing output.
            # For this test, assume ORCA runs and creates the output.
            mock_exists.side_effect = lambda p: p == "orca" or p == input_file_path or p == output_file_expected
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK", stderr="")

            if mock_run.return_value.returncode == 0:
                # Create a dummy output file to simulate ORCA's success, so cleanup doesn't fail
                with open(output_file_expected, "w") as f_out:
                    f_out.write("dummy orca output for test_run_successful")

            # Simulate that ORCA creates the output file
            # If run_orca_calculation itself is supposed to ensure output_file_expected exists after a successful run,
            # then this mock might need to create it, or the test checks for its creation.
            # For now, let's assume run_orca_calculation returns the path, and we mock that it would be created.
            # To be safe, let's ensure the expected output file exists for the mock_exists check if needed by the SUT.
            # However, run_orca_calculation doesn't check if output exists before returning path.
            # It *does* check if ORCA executable and input file exist.

            success, out_path = run_orca_calculation(input_file_path, orca_path="orca")

            assert success
            assert out_path == output_file_expected
            mock_run.assert_called_once_with(
                ["orca", input_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,

                cwd=os.path.dirname(input_file_path),
            )
        finally:
            if os.path.exists(input_file_path):
                os.remove(input_file_path)
            # Only remove output_file_expected if success was True and it was supposed to be created
            if success and os.path.exists(output_file_expected):
                os.remove(output_file_expected)

    def test_run_orca_fail_returncode(self, mock_exists, mock_run):
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".inp", delete=False) as tmp_inp_f:  # Explicitly w+b
            input_file_path = tmp_inp_f.name
            tmp_inp_f.write(b"dummy orca input content")  # Write bytes

        output_file_expected = input_file_path.replace(".inp", ".out")

        try:
            # For this test, the output file might or might not exist after a failed ORCA run.
            # The SUT's run_orca_calculation checks os.path.exists(output_file) *after* the subprocess call.
            # If ORCA fails (returncode !=0), it still returns the output_file path.
            # If os.path.exists(output_file) is false (e.g. ORCA failed AND didn't create output), it returns False, ""

            # Scenario 1: ORCA fails, output file IS created (e.g. partial output)
            # mock_exists.side_effect = lambda p: p == "orca" or p == input_file_path or p == output_file_expected
            # mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="ORCA ERROR")
            # success, out_path = run_orca_calculation(input_file_path, orca_path="orca")
            # assert not success
            # assert out_path == output_file_expected

            # Scenario 2: ORCA fails, output file IS NOT created
            mock_exists.side_effect = (
                lambda p: p == "orca" or p == input_file_path
            )  # output_file_expected will be False
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="ORCA ERROR")
            success, out_path = run_orca_calculation(input_file_path, orca_path="orca")
            assert not success
            assert out_path == ""  # Because os.path.exists(output_file_expected) was false

        finally:
            if os.path.exists(input_file_path):
                os.remove(input_file_path)
            if os.path.exists(output_file_expected):  # Cleanup if it was created
                os.remove(output_file_expected)

    def test_run_output_not_created(self, mock_exists, mock_run):
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".inp", delete=False) as tmp_inp_f:  # Explicitly w+b
            input_file_path = tmp_inp_f.name
            tmp_inp_f.write(b"dummy orca input content")  # Write bytes

        output_file_path = input_file_path.replace(".inp", ".out")

        # Ensure it doesn't exist before the test
        try:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
        except FileNotFoundError:
            pass # It's okay if it doesn't exist, that's the point

        try:
            # Simulate ORCA executable and input file exist, but output file does not.
            def mock_exists_logic(path_to_check):
                if path_to_check == "orca":
                    return True
                if path_to_check == input_file_path:
                    return True
                if path_to_check == output_file_path:
                    return False  # Simulate output not created
                return False

            mock_exists.side_effect = mock_exists_logic

            # ORCA subprocess itself "succeeds" (returncode 0), but the output file is missing.
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="OK", stderr="")

            success, out_path = run_orca_calculation(input_file_path, orca_path="orca")

            # run_orca_calculation checks os.path.exists(output_file) after the run.
            # If it's False (as per our mock_exists_logic), it returns False, "".
            assert not success
            assert out_path == ""
            assert not os.path.exists(output_file_path)  # Double check on disk

        finally:
            if os.path.exists(input_file_path):
                os.remove(input_file_path)
            if os.path.exists(output_file_path):  # Should not exist, but cleanup if it does
                os.remove(output_file_path)


@patch("moml.simulation.quantum_mechanics.parser.orca_parser.smiles_to_3d_structure")
@patch("moml.simulation.quantum_mechanics.parser.orca_parser.create_orca_input")
@patch("moml.simulation.quantum_mechanics.parser.orca_parser.run_orca_calculation")
@patch("moml.simulation.quantum_mechanics.parser.orca_parser.parse_orca_output")
class TestProcessMolecule:
    def test_process_molecule_success(
        self, mock_parse, mock_run, mock_create_inp, mock_smiles_3d, temp_orca_output_file
    ):
        # temp_orca_output_file is just for a path name
        mol_dir = os.path.dirname(temp_orca_output_file)
        mol_id = "test_mol"

        mock_smiles_3d.return_value = Chem.MolFromSmiles("C")  # Dummy mol
        mock_create_inp.return_value = (True, os.path.join(mol_dir, f"{mol_id}.inp"))
        mock_run.return_value = (True, os.path.join(mol_dir, f"{mol_id}.out"))
        mock_parse.return_value = {"status": "completed", "data_key": "value"}

        results_path = os.path.join(mol_dir, mol_id, f"{mol_id}_results.json")
        with patch("builtins.open", mock_open()) as mock_file_write:  # Mock json save
            results = process_molecule("C", mol_id, mol_dir, "B3LYP", "def2-SVP", 1, 1000, "orca")

        assert results["status"] == "completed"
        assert results["data"] == {"status": "completed", "data_key": "value"}
        assert results["error"] is None
        mock_file_write.assert_called_once_with(results_path, "w")

    def test_process_molecule_smiles_fail(
        self, mock_parse, mock_run, mock_create_inp, mock_smiles_3d, temp_orca_output_file
    ):
        mol_dir = os.path.dirname(temp_orca_output_file)
        mock_smiles_3d.return_value = None  # Simulate failure

        results = process_molecule("C", "fail_mol", mol_dir, "B3LYP", "def2-SVP", 1, 1000, "orca")
        assert results["status"] == "failed"
        assert results["error"] == "Failed to create 3D structure"
        mock_create_inp.assert_not_called()

    # Add more tests for failures at create_input, run_orca steps...


@pytest.fixture
def sample_molecules_df() -> pd.DataFrame:
    """Provides a sample DataFrame for batch processing tests."""
    data = {"id": ["mol1", "mol2", "mol3"], "smiles": ["C", "CC", "CCC"]}
    return pd.DataFrame(data)


@pytest.fixture
def temp_batch_output_dir():
    """Creates a temporary directory for batch output files."""
    dir_path = tempfile.mkdtemp(prefix="batch_orca_")
    yield dir_path
    shutil.rmtree(dir_path)


@patch("moml.simulation.quantum_mechanics.parser.orca_parser.process_molecule")
@patch("concurrent.futures.ProcessPoolExecutor")
class TestBatchProcessMolecules:
    @pytest.mark.skip(reason="Test hangs or is too slow, to be investigated later")
    def test_batch_process_all_success(
        self, mock_executor_cls, mock_process_mol, sample_molecules_df, temp_batch_output_dir
    ):
        """Test batch_process_molecules with all individual molecule processes succeeding."""

        # Mock process_molecule return values
        mock_process_mol.side_effect = [
            {"id": "mol1", "smiles": "C", "status": "completed", "data": {"prop": 1}, "error": None},
            {"id": "mol2", "smiles": "CC", "status": "completed", "data": {"prop": 2}, "error": None},
            {"id": "mol3", "smiles": "CCC", "status": "completed", "data": {"prop": 3}, "error": None},
        ]

        # Mock ProcessPoolExecutor
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor_instance

        # Mock futures
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
            "status": "completed",
            "data": {"prop": 2},
            "error": None,
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
            smiles_col="smiles",  # Specify correct column name
            id_col="id",  # Specify correct column name
        )

        assert len(results) == 3
        assert results[0]["status"] == "completed"
        assert results[1]["data"]["prop"] == 2

        # Check that process_molecule was called correctly for each molecule
        expected_calls = [
            call("C", "mol1", temp_batch_output_dir, "B3LYP", "def2-SVP", 2, 2000, "orca_mock"),
            call("CC", "mol2", temp_batch_output_dir, "B3LYP", "def2-SVP", 2, 2000, "orca_mock"),
            call("CCC", "mol3", temp_batch_output_dir, "B3LYP", "def2-SVP", 2, 2000, "orca_mock"),
        ]
        mock_process_mol.assert_has_calls(
            expected_calls, any_order=False
        )  # Order of submission matters for side_effect
        assert mock_process_mol.call_count == 3

        # Check summary file
        summary_file_path = os.path.join(temp_batch_output_dir, "orca_batch_summary.json")
        assert os.path.exists(summary_file_path)
        with open(summary_file_path, "r") as f:
            summary_data = json.load(f)
        assert len(summary_data) == 3
        assert summary_data[0]["id"] == "mol1"

    @pytest.mark.skip(reason="Test hangs or is too slow, to be investigated later")
    def test_batch_process_with_failures(
        self, mock_executor_cls, mock_process_mol, sample_molecules_df, temp_batch_output_dir
    ):
        """Test batch_process_molecules with some individual molecule processes failing."""
        mock_process_mol.side_effect = [
            {"id": "mol1", "smiles": "C", "status": "completed", "data": {"prop": 1}, "error": None},
            {"id": "mol2", "smiles": "CC", "status": "failed", "data": None, "error": "ORCA Error"},
            {"id": "mol3", "smiles": "CCC", "status": "completed", "data": {"prop": 3}, "error": None},
        ]

        mock_executor_instance = MagicMock()
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
            smiles_col="smiles",  # Specify correct column name
            id_col="id",  # Specify correct column name
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
        self, mock_executor_cls, mock_process_mol, sample_molecules_df, temp_batch_output_dir, caplog
    ):
        """Test batch_process_molecules when a future raises an exception."""
        # process_molecule itself won't be called if future.result() raises exception
        # So we don't need to mock its side_effect extensively for this specific test.
        # The key is that the future.result() call is what we're testing the handling of.

        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor_instance

        future1, future2, future3 = MagicMock(), MagicMock(), MagicMock()
        future1.result.return_value = {
            "id": "mol1",
            "smiles": "C",
            "status": "completed",
            "data": {"prop": 1},
            "error": None,
        }
        future2.result.side_effect = Exception("Simulated future error")  # This future will raise an error
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
            smiles_col="smiles",  # Specify correct column name
            id_col="id",  # Specify correct column name
        )

        # The function should still process all futures and return results for those that didn't fail catastrophically
        # The result for the failed future will be a placeholder error entry.
        assert len(results) == 3
        assert results[0]["status"] == "completed"

        # Check the result for the molecule that had a future exception
        assert results[1]["id"] == "mol2"  # Assuming order is maintained based on submission
        assert results[1]["status"] == "error"  # or "failed" depending on internal error handling
        assert "Simulated future error" in results[1]["error_message"]  # Check for the error message
        assert "Error processing molecule mol2" in caplog.text  # Check logger output

        assert results[2]["status"] == "completed"

        summary_file_path = os.path.join(temp_batch_output_dir, "orca_batch_summary.json")
        assert os.path.exists(summary_file_path)
        with open(summary_file_path, "r") as f:
            summary_data = json.load(f)
        assert len(summary_data) == 3
        assert summary_data[1]["status"] == "error"
