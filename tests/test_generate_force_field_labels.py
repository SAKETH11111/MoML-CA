"""
tests/test_generate_force_field_labels.py

Unit tests for the force field label generation script.
"""

from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.generate_force_field_labels import generate_labels, load_smiles_map


def test_load_smiles_map(tmp_path: Path) -> None:
    """
    Test loading a SMILES map from a CSV file.
    
    Args:
        tmp_path (Path): Temporary directory path provided by pytest.
    """
    csv_path = tmp_path / "mols.csv"
    df = pd.DataFrame({"DTXSID": ["1", "2"], "SMILES": ["C", "CC"]})
    df.to_csv(csv_path, index=False)
    result = load_smiles_map(str(csv_path))
    assert result == {"1": "C", "2": "CC"}


@patch("scripts.generate_force_field_labels.ForceFieldMapper")
@patch("scripts.generate_force_field_labels.safe_parse_orca_output")
def test_generate_labels(mock_parse: MagicMock, mock_mapper_cls: MagicMock, tmp_path: Path) -> None:
    """
    Test label generation using mocked ForceFieldMapper and ORCA parser.
    
    Args:
        mock_parse (MagicMock): Mock of the ORCA parser function.
        mock_mapper_cls (MagicMock): Mock of the ForceFieldMapper class.
        tmp_path (Path): Temporary directory path provided by pytest.
    """
    out_file = tmp_path / "mol1.out"
    out_file.write_text("dummy")

    mock_parse.return_value = {
        "mulliken_charges": [0.1],
        "optimized_geometry": [{"symbol": "H", "coordinates": [0.0, 0.0, 0.0]}],
    }
    mock_mapper = MagicMock()
    mock_mapper.generate_force_field_parameters.return_value = {"charges": {"H1": 0.1}}
    mock_mapper_cls.return_value = mock_mapper

    labels = generate_labels(str(tmp_path), {"mol1": "[H]"})

    mock_parse.assert_called_with(out_file)
    mock_mapper.generate_force_field_parameters.assert_called_once()
    assert labels == {
        "mol1": {
            "charges": {"H1": 0.1},
            "bonds": {},
            "angles": {},
            "dihedrals": {},
        }
    }
