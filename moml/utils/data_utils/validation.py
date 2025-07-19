"""
moml/utils/data_utils/validation.py

This module provides utility functions for validating chemical data, with a
primary focus on handling and canonicalizing SMILES (Simplified Molecular-Input
Line-Entry System) strings using the RDKit library.
"""

import logging
from typing import Optional, Tuple

from rdkit import Chem

# Logger for this module
logger = logging.getLogger(__name__)


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str], Optional[Chem.Mol], Optional[str]]:
    """
    Validate a SMILES string and convert it to its canonical form.

    This function checks if a given SMILES string is chemically valid and, if so,
    returns its canonical representation along with the corresponding RDKit
    molecule object.

    Args:
        smiles (str): The SMILES string to be validated.

    Returns:
        Tuple[bool, Optional[str], Optional[Chem.Mol], Optional[str]]: A tuple
        containing four elements:
        - A boolean flag indicating if the SMILES string is valid.
        - The canonical SMILES string if validation is successful; otherwise, None.
        - The RDKit molecule object if validation is successful; otherwise, None.
        - An error message string if validation fails; otherwise, None.
    """
    if not isinstance(smiles, str) or not smiles:
        return False, None, None, "SMILES input must be a non-empty string."

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, None, f"Invalid SMILES notation: '{smiles}'"

        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return True, canonical_smiles, mol, None

    except Exception as e:
        logger.error(f"An unexpected error occurred while validating SMILES '{smiles}': {e}")
        return False, None, None, f"Exception during SMILES processing: {e}"
