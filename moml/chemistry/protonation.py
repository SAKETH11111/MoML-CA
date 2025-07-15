"""
Assign the major protonation/tautomer state for PFAS (and, later, other solutes)
given an env_vec pH.  Returns an RDKit Mol with explicit hydrogens and the total
integer charge; downstream mapper picks the partial charges from MGNN or AM1-BCC.
"""

from typing import Tuple
from rdkit import Chem

# Quick-and-dirty lookup; extend as needed.
_PFAS_LOOKUP = {
    "PFBA": {"pKa": 3.8, "smiles_acid": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFOA": {"pKa": 3.3, "smiles_acid": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFOS": {"pKa": -3.0, "smiles_acid": "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFBS": {"pKa": -3.0, "smiles_acid": "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFHxA": {"pKa": 3.0, "smiles_acid": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFNA": {"pKa": 3.0, "smiles_acid": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFDA": {"pKa": 3.0, "smiles_acid": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFHpA": {"pKa": 3.0, "smiles_acid": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFHxS": {"pKa": -3.0, "smiles_acid": "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
    "PFDS": {"pKa": -3.0, "smiles_acid": "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"},
}

def protonate(smiles: str, pH: float) -> Tuple[Chem.Mol, int]:
    """
    Determine the protonation state of a molecule based on pH.
    
    Args:
        smiles: SMILES string of the molecule
        pH: Environmental pH value
        
    Returns:
        Tuple containing:
        - RDKit molecule with explicit hydrogens
        - Integer net charge
    """
    input_mol = Chem.MolFromSmiles(smiles)
    if input_mol is None:
        raise ValueError(f"Invalid SMILES string provided: {smiles}")

    for name, data in _PFAS_LOOKUP.items():
        acid_smiles = data["smiles_acid"]
        acid_mol = Chem.MolFromSmiles(acid_smiles)
        if acid_mol is None:
            # This should ideally not happen if _PFAS_LOOKUP is well-defined
            continue

        # Use substructure matching for more robust identification
        if input_mol.HasSubstructMatch(acid_mol):
            mol = Chem.MolFromSmiles(acid_smiles) # Use the acid form from lookup
            if mol is None: # Should not be None if acid_mol was not None
                continue

            if pH > data["pKa"] + 1.0:                             # deprotonated
                Chem.SanitizeMol(mol)
                mol = Chem.RemoveHs(mol)
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() == 1:
                        atom.SetFormalCharge(-1)
                        atom.SetNumExplicitHs(0)
                        break
                return Chem.AddHs(mol), -1
            else:                                                  # protonated
                return Chem.AddHs(mol), 0
    # fallback: just return original
    # mol is already input_mol from the initial check
    return Chem.AddHs(input_mol), Chem.GetFormalCharge(input_mol)