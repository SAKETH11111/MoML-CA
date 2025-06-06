import torch
import numpy as np
import h5py
import os
import logging
from typing import List
from torch_geometric.data import InMemoryDataset, Data

logger = logging.getLogger(__name__)


class SpiceDataset(InMemoryDataset):
    """
    SPICE dataset for molecular property prediction with node-level forces and graph-level energies.
    
    This dataset processes raw HDF5 data into PyTorch Geometric Data objects,
    storing per-atom forces as `node_y` and total energy as `y_graph`.
    It inherits from `InMemoryDataset` for efficient loading of processed data.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): The dataset split ('train', 'val', 'test'). Defaults to "train".
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(self, root: str, split: str = "train", transform=None):
        self.split = split
        super().__init__(root, transform, None) # pre_transform is None. This will trigger download/process if needed.
        
        # Explicitly load data if processed file exists and is not empty
        if os.path.exists(self.processed_paths[0]):
            try:
                loaded_data = torch.load(self.processed_paths[0])
                if loaded_data[0] is not None: # Check if the saved data is not None (for empty datasets)
                    self.data, self.slices = loaded_data
                else:
                    self.data, self.slices = None, None # Handle case where (None, None) was saved
            except Exception as e:
                logger.warning(f"Could not load processed data from {self.processed_paths[0]}: {e}")
                self.data, self.slices = None, None # Fallback to empty
        else:
            self.data, self.slices = None, None # No processed file, so data is empty initially

    @property
    def raw_file_names(self) -> List[str]:
        """Return the names of the raw files in the dataset."""
        return ["SPICE-1.1.4.hdf5"]

    @property
    def processed_file_names(self) -> List[str]:
        """Return the names of the processed files."""
        return [f"{self.split}.pt"]

    def download(self):
        """
        Download the dataset from the web.
        
        This method is a placeholder as the SPICE dataset is assumed to be
        already present in the `data/spice` directory.
        """
        # nothing – you already dropped the file in data/spice/
        pass

    def _get_edge_index(self, positions: torch.Tensor, cutoff: float = 3.0) -> torch.Tensor:
        """
        Create edge indices based on distance cutoff.
        
        Args:
            positions (torch.Tensor): Tensor of atomic coordinates [N_atoms, 3].
            cutoff (float): Distance cutoff for creating edges in Angstroms.
                            Defaults to 3.0.
        
        Returns:
            torch.Tensor: Edge index tensor [2, num_edges].
        """
        n_atoms = positions.shape[0]
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions)
        
        # Create edges for atoms within cutoff distance (excluding self-loops)
        mask = (dist_matrix < cutoff) & (dist_matrix > 0)
        edge_index = mask.nonzero().t()
        
        # Ensure we have at least some edges (fallback to nearest neighbors if no edges)
        if edge_index.shape[1] == 0 and n_atoms > 1:
            # Connect each atom to its nearest neighbor
            # This is a fallback and might not be chemically accurate for all cases
            for i in range(n_atoms):
                distances = dist_matrix[i].clone()
                distances[i] = float('inf')  # Exclude self
                nearest = torch.argmin(distances)
                # Add bidirectional edges
                edge_index = torch.cat([
                    edge_index,
                    torch.tensor([[i, nearest], [nearest, i]], dtype=torch.long).t()
                ], dim=1)
            # Remove duplicate edges if any
            edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index

    def process(self):
        """
        Process the raw HDF5 data into PyTorch Geometric Data objects.
        
        Each conformer from the HDF5 file is converted into a Data object
        with atomic numbers (z), coordinates (pos), total energy (y_graph),
        and atomic forces (node_y). Edge indices are generated based on
        a distance cutoff.
        """
        raw_file_path = self.raw_paths[0]
        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"Raw SPICE dataset not found at {raw_file_path}")

        h5 = h5py.File(raw_file_path, "r")
        X: List[Data] = []

        mol_keys = list(h5.keys())
        logger.debug(f"Found mol_keys: {mol_keys}")
        for mol_key in mol_keys:
            mol_data = h5[mol_key]
            atomic_numbers = mol_data['atomic_numbers'][:]
            
            conformation_keys = sorted(mol_data['conformations'].keys(), key=int)
            logger.debug(f"For {mol_key}, found conformation_keys: {conformation_keys}")
            for conf_idx_str in conformation_keys:
                coords  = np.array(mol_data['conformations'][conf_idx_str])  # [N,3]
                # Use the integer index for gradient and energy as they are datasets
                conf_int_idx = int(conf_idx_str)
                forces  = np.array(mol_data['dft_total_gradient'][conf_int_idx]) # [N,3]
                energy  = np.array(mol_data['dft_total_energy'][conf_int_idx]).item() # scalar

                # Generate edge_index
                pos_tensor = torch.tensor(coords, dtype=torch.float32)
                edge_index = self._get_edge_index(pos_tensor)

                d = Data(
                    pos=pos_tensor,
                    z=torch.tensor(atomic_numbers, dtype=torch.long),
                    y_graph=torch.tensor([energy], dtype=torch.float32),
                    node_y=torch.tensor(forces, dtype=torch.float32),
                    edge_index=edge_index
                )
                X.append(d)
        
        h5.close() 
        
        logger.debug(f"Total samples collected (len(X)): {len(X)}")

        if not X: # Handle empty dataset case
            logger.debug("No data collected, saving empty dataset (None, None).")
            torch.save((None, None), self.processed_paths[0]) # Save (None, None) for empty dataset
            return

        data, slices = self.collate(X)
        logger.debug(f"Collated data: {data}")
        logger.debug(f"Collated slices: {slices}")
        torch.save((data, slices), self.processed_paths[0])
