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
    for name, data in _PFAS_LOOKUP.items():
        if smiles.startswith(data["smiles_acid"][:8]):            # crude match
            mol = Chem.MolFromSmiles(data["smiles_acid"])
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
    mol = Chem.MolFromSmiles(smiles)
    return Chem.AddHs(mol), Chem.GetFormalCharge(mol) 