from rdkit import Chem


def validate_smiles(smiles: str) -> tuple:
    """
    Validate a SMILES string and convert to canonical form.

    Args:
        smiles: The SMILES string to validate

    Returns:
        Tuple containing:
            - Boolean indicating if SMILES is valid
            - Canonical SMILES (if valid, otherwise None)
            - RDKit Mol object (if valid and conversion successful, otherwise None)
            - Error message (if invalid or error, otherwise None)
    """
    if not smiles or not isinstance(smiles, str):
        return False, None, None, "Empty or non-string SMILES input"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, None, f"Invalid SMILES: {smiles}"
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return True, canonical_smiles, mol, None
    except Exception as e:
        return False, None, None, f"Error processing SMILES: {str(e)}"
