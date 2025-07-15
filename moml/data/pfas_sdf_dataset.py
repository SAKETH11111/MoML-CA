import os
import glob
from typing import Callable, List, Optional
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen # Added Crippen

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class PFASSDFDataset(InMemoryDataset):
    """
    Dataset for PFAS molecules from SDF files.
    Computes 19-dimensional molecular descriptors as graph-level targets.
    """
    
    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Explicitly load data if processed file exists
        if os.path.exists(self.processed_paths[0]):
            try:
                loaded_data = torch.load(self.processed_paths[0])
                if loaded_data[0] is not None:
                    self.data, self.slices = loaded_data
                else:
                    self.data, self.slices = None, None
            except Exception as e:
                print(f"Could not load processed PFAS data from {self.processed_paths[0]}: {e}")
                self.data, self.slices = None, None
        else:
            self.data, self.slices = None, None
        
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of SDF files in the raw directory."""
        sdf_files = glob.glob(os.path.join(self.raw_dir, "*.sdf"))
        return [os.path.basename(f) for f in sdf_files]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return processed file names."""
        split_suffix = f"_{self.split}" if self.split else ""
        return [f"pfas_sdf{split_suffix}.pt"]
    
    def download(self):
        """No download needed - SDF files should already be present."""
        if not self.raw_file_names:
            raise FileNotFoundError(f"No SDF files found in {self.raw_dir}")
    
    def process(self):
        """Process SDF files into PyTorch Geometric Data objects."""
        data_list = []
        
        for sdf_file in self.raw_file_names:
            sdf_path = os.path.join(self.raw_dir, sdf_file)
            
            try:
                # Read molecules from SDF file
                suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
                
                for mol in suppl:
                    if mol is None:
                        continue
                        
                    # Convert molecule to Data object
                    data = self._mol_to_data(mol)
                    if data is not None:
                        data_list.append(data)
                        
            except Exception as e:
                print(f"Error processing {sdf_file}: {e}")
                continue
        
        print(f"Successfully processed {len(data_list)} PFAS molecules")
        
        # Apply pre_filter and pre_transform if specified
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Save processed data
        if not data_list:
            # Handle empty dataset
            torch.save((None, None), self.processed_paths[0])
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _mol_to_data(self, mol: Chem.Mol) -> Optional[Data]:
        """Convert RDKit molecule to PyTorch Geometric Data object."""
        try:
            # Get atom features
            atoms = mol.GetAtoms()
            N = len(atoms)
            
            if N == 0:
                return None
            
            # Node features: atomic numbers
            z = torch.tensor([atom.GetAtomicNum() for atom in atoms], dtype=torch.long)
            
            # Node positions (3D coordinates)
            try:
                conf = mol.GetConformer()
                pos = torch.tensor([[conf.GetAtomPosition(i).x,
                                   conf.GetAtomPosition(i).y, 
                                   conf.GetAtomPosition(i).z] for i in range(N)], 
                                 dtype=torch.float)
            except (AttributeError, ValueError):
                # If no conformer, generate 2D coordinates
                from rdkit.Chem import rdDepictor
                rdDepictor.Compute2DCoords(mol)
                conf = mol.GetConformer()
                pos = torch.tensor([[conf.GetAtomPosition(i).x,
                                   conf.GetAtomPosition(i).y, 
                                   0.0] for i in range(N)], 
                                 dtype=torch.float)
            
            # Edge indices (bonds)
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])  # Add both directions
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                # No bonds - create empty edge_index
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # Compute 19-dimensional molecular descriptors as targets
            y = self._compute_molecular_descriptors(mol)
            
            return Data(z=z, pos=pos, edge_index=edge_index, y=y)
            
        except Exception as e:
            print(f"Error converting molecule to data: {e}")
            return None
    
    def _compute_molecular_descriptors(self, mol: Chem.Mol) -> torch.Tensor:
        """Compute 19 molecular descriptors to match QM9 target dimensions."""
        try:
            # Basic molecular properties (similar to QM9)
            descriptors = [
                Descriptors.MolWt(mol),                    # Molecular weight
                Descriptors.ExactMolWt(mol),               # Exact molecular weight  
                Chem.Crippen.MolLogP(mol),                      # LogP
                Descriptors.TPSA(mol),                     # Topological polar surface area
                Descriptors.NumHAcceptors(mol),            # H-bond acceptors
                Descriptors.NumHDonors(mol),               # H-bond donors
                Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
                Descriptors.NumAromaticRings(mol),         # Aromatic rings
                Descriptors.NumSaturatedRings(mol),        # Saturated rings
                mol.GetNumHeavyAtoms(),                    # Number of heavy atoms (instead of FractionCsp3)
                Descriptors.BalabanJ(mol),                 # Balaban J index
                Descriptors.BertzCT(mol),                  # Bertz complexity
                Descriptors.HallKierAlpha(mol),            # Hall-Kier alpha
                Descriptors.Kappa1(mol),                   # Kappa shape index 1
                Descriptors.Kappa2(mol),                   # Kappa shape index 2
                Descriptors.Kappa3(mol),                   # Kappa shape index 3
                Descriptors.LabuteASA(mol),                # Labute ASA
                Descriptors.NumHeteroatoms(mol),           # Number of heteroatoms
                len([a for a in mol.GetAtoms() if a.GetSymbol() == 'F'])  # Fluorine count (PFAS-specific)
            ]
            
            # Handle any None or invalid values
            descriptors = [float(d) if d is not None and not np.isnan(float(d)) else 0.0 
                          for d in descriptors]
            
            return torch.tensor(descriptors, dtype=torch.float)
            
        except Exception as e:
            print(f"Error computing molecular descriptors: {e}")
            # Return zeros if computation fails
            return torch.zeros(19, dtype=torch.float)
    
