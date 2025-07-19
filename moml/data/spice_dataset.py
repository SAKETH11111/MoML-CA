"""
moml/data/spice_dataset.py

SPICE dataset for quantum mechanical molecular property prediction.

This module provides a PyTorch Geometric dataset class for loading and processing
the SPICE (Submolecular Property Informed Chemical Embeddings) dataset. The SPICE
dataset contains quantum mechanical calculations of molecular properties including
energies and forces, making it suitable for training machine learning models on
accurate quantum chemical data.

Classes:
    SpiceDataset: Dataset class for SPICE molecular data with energies and forces

Example:
    Basic usage for loading SPICE dataset:
    
    >>> from moml.data.spice_dataset import SpiceDataset
    >>> dataset = SpiceDataset(root='data/spice/', split='train')
    >>> print(f"Dataset size: {len(dataset)}")
    >>> graph = dataset[0]
    >>> print(f"Energy: {graph.y_graph}")
    >>> print(f"Forces: {graph.node_y.shape}")
"""

import logging
import os
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

# Constants
DEFAULT_SPLIT = "train"
DEFAULT_CUTOFF = 3.0
MAX_MOLECULES_LIMIT = 100  # Limit for testing/development
RAW_FILENAME = "SPICE-1.1.4.hdf5"

# Configure module logger
logger = logging.getLogger(__name__)

try:
    import h5py  # type: ignore
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py not available. SPICE dataset functionality will be limited.")
    
    # Create dummy h5py for fallback
    class h5py:  # type: ignore
        """Dummy h5py module when not available."""
        
        @staticmethod
        def File(filepath, mode):
            """Dummy File class."""
            raise ImportError("h5py is required for SPICE dataset")


class SpiceDataset(InMemoryDataset):
    """
    SPICE dataset for molecular property prediction with quantum mechanical data.
    
    This dataset processes the SPICE HDF5 data into PyTorch Geometric Data objects,
    storing per-atom forces as node-level targets (`node_y`) and total molecular
    energy as graph-level targets (`y_graph`). The dataset includes atomic
    coordinates, atomic numbers, and edge connectivity based on distance cutoffs.
    
    The SPICE dataset provides high-quality quantum mechanical calculations for
    diverse molecular systems, making it ideal for training models on accurate
    energies and forces for molecular property prediction tasks.

    Args:
        root: Root directory where the dataset should be saved
        split: Dataset split identifier ('train', 'val', 'test')
        transform: Optional transform applied to each graph during loading
        pre_transform: Optional transform applied during processing
    
    Attributes:
        split: Dataset split identifier
        data: Processed graph data
        slices: Data slicing information for batching
    
    Example:
        >>> dataset = SpiceDataset(root='data/spice/', split='train')
        >>> print(f"Number of conformers: {len(dataset)}")
        >>> graph = dataset[0]
        >>> print(f"Atomic numbers: {graph.z}")
        >>> print(f"Coordinates: {graph.pos}")
        >>> print(f"Energy: {graph.y_graph}")
        >>> print(f"Forces: {graph.node_y}")
    
    Note:
        The raw SPICE HDF5 file should be placed in the 'raw' subdirectory
        of the root path. Processed data will be saved to the 'processed'
        subdirectory for faster subsequent loading.
    
    Raises:
        ImportError: If h5py is not available
        FileNotFoundError: If the raw SPICE file is not found
    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = DEFAULT_SPLIT,
        transform: Optional[Callable[..., Any]] = None,
        pre_transform: Optional[Callable[..., Any]] = None
    ) -> None:
        """
        Initialize the SPICE dataset.
        
        Args:
            root: Root directory path
            split: Dataset split identifier
            transform: Optional transform function
            pre_transform: Optional pre-processing transform
            
        Raises:
            ImportError: If h5py is not available
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for SpiceDataset")
        
        self.split = split
        super().__init__(root, transform, pre_transform)
        
        # Load processed data if available
        self._load_processed_data()

    def _load_processed_data(self) -> None:
        """
        Load processed data from disk if available.
        
        Attempts to load pre-processed graph data from the processed directory.
        If loading fails or no processed data exists, initializes empty data.
        """
        processed_file = self.processed_paths[0]
        
        if os.path.exists(processed_file):
            try:
                loaded_data = torch.load(processed_file, weights_only=True)
                
                # Check if data is valid (not None for empty datasets)
                if loaded_data[0] is not None:
                    self.data, self.slices = loaded_data
                    logger.info(f"Loaded processed SPICE data from {processed_file}")
                else:
                    self.data, self.slices = None, None
                    logger.warning("Processed data file contains None values")
                    
            except Exception as e:
                logger.warning(f"Could not load processed data from {processed_file}: {e}")
                self.data, self.slices = None, None
        else:
            self.data, self.slices = None, None
            logger.debug("No processed data file found")

    @property
    def raw_file_names(self) -> List[str]:
        """
        Return the names of the raw files in the dataset.
        
        Returns:
            List containing the SPICE HDF5 filename
        """
        return [RAW_FILENAME]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Return the names of the processed files based on split.
        
        Returns:
            List containing the processed file name for the current split
        """
        return [f"{self.split}.pt"]

    def download(self) -> None:
        """
        Download placeholder for the SPICE dataset.
        
        This method is a placeholder since the SPICE dataset is assumed to be
        manually placed in the raw directory. The SPICE dataset should be
        downloaded separately and placed as 'SPICE-1.1.4.hdf5' in the raw
        data directory.
        
        Note:
            The SPICE dataset can be obtained from the official SPICE repository
            or data portal. Place the HDF5 file in the 'raw' subdirectory.
        """
        logger.info(
            f"No automatic download available. Please place {RAW_FILENAME} "
            f"in {self.raw_dir}"
        )

    def process(self) -> None:
        """
        Process the raw HDF5 data into PyTorch Geometric Data objects.
        
        This method reads the SPICE HDF5 file and converts each molecular
        conformer into a graph representation with atomic features, coordinates,
        energies, and forces. Edge connectivity is generated based on distance
        cutoffs between atoms.
        
        The processing includes:
        - Reading atomic numbers, coordinates, energies, and forces
        - Generating edge indices based on spatial proximity
        - Creating Data objects with appropriate target assignments
        - Applying pre-transforms if specified
        - Saving processed data for future use
        
        Raises:
            FileNotFoundError: If the raw SPICE file is not found
        """
        raw_file_path = self.raw_paths[0]
        
        # Validate raw file exists
        if not os.path.exists(raw_file_path):
            error_msg = f"Raw SPICE dataset not found at {raw_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Processing SPICE dataset from {raw_file_path}")
        
        try:
            # Open HDF5 file and process data
            with h5py.File(raw_file_path, "r") as h5_file:  # type: ignore
                data_list = self._process_molecules(h5_file)
            
            # Apply pre-transforms if specified
            if self.pre_transform is not None:
                logger.info("Applying pre-transforms")
                data_list = [self.pre_transform(data) for data in data_list]
            
            # Save processed data
            self._save_processed_data(data_list)
            
        except Exception as e:
            logger.error(f"Error processing SPICE dataset: {e}")
            raise

    def _process_molecules(self, h5_file) -> List[Data]:
        """
        Process molecules from HDF5 file into Data objects.
        
        Args:
            h5_file: Opened HDF5 file object
            
        Returns:
            List of processed Data objects
        """
        data_list = []
        
        # Determine the structure of the HDF5 file
        mol_container = h5_file['molecules'] if 'molecules' in h5_file else h5_file
        
        mol_keys = list(mol_container.keys())
        logger.info(f"Found {len(mol_keys)} molecules in dataset")
        
        # Limit molecules for development/testing
        if len(mol_keys) > MAX_MOLECULES_LIMIT:
            mol_keys = mol_keys[:MAX_MOLECULES_LIMIT]
            logger.warning(f"Limited to {MAX_MOLECULES_LIMIT} molecules for processing")
        
        total_conformers = 0
        
        for mol_key in mol_keys:
            try:
                mol_data = mol_container[mol_key]
                conformers = self._process_single_molecule(mol_data, mol_key)
                data_list.extend(conformers)
                total_conformers += len(conformers)
                
            except Exception as e:
                logger.warning(f"Error processing molecule {mol_key}: {e}")
                continue
        
        logger.info(f"Processed {total_conformers} conformers from {len(mol_keys)} molecules")
        return data_list

    def _process_single_molecule(self, mol_data, mol_key: str) -> List[Data]:
        """
        Process a single molecule's conformers.
        
        Args:
            mol_data: HDF5 group containing molecule data
            mol_key: Molecule identifier for logging
            
        Returns:
            List of Data objects for each conformer
        """
        conformers = []
        
        try:
            # Extract atomic numbers (same for all conformers)
            atomic_numbers = mol_data['atomic_numbers'][:]
            
            # Get number of conformers
            num_conformers = mol_data['dft_total_energy'].shape[0]
            logger.debug(f"Molecule {mol_key}: {num_conformers} conformers")

            # Process each conformer
            for conf_idx in range(num_conformers):
                conformer = self._process_conformer(
                    mol_data, conf_idx, atomic_numbers
                )
                if conformer is not None:
                    conformers.append(conformer)
                    
        except Exception as e:
            logger.warning(f"Error processing molecule {mol_key}: {e}")
        
        return conformers

    def _process_conformer(
        self, 
        mol_data, 
        conf_idx: int, 
        atomic_numbers: np.ndarray
    ) -> Optional[Data]:
        """
        Process a single conformer into a Data object.
        
        Args:
            mol_data: HDF5 group containing molecule data
            conf_idx: Conformer index
            atomic_numbers: Array of atomic numbers
            
        Returns:
            Data object or None if processing fails
        """
        try:
            # Extract conformer data
            coordinates = np.array(mol_data['conformations'][conf_idx])
            forces = np.array(mol_data['dft_total_gradient'][conf_idx])
            energy = np.array(mol_data['dft_total_energy'][conf_idx]).item()

            # Convert to tensors
            pos_tensor = torch.tensor(coordinates, dtype=torch.float32)
            z_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
            forces_tensor = torch.tensor(forces, dtype=torch.float32)
            energy_tensor = torch.tensor([energy], dtype=torch.float32)
            
            # Generate edge connectivity
            edge_index = self._get_edge_index(pos_tensor)

            # Create Data object
            return Data(
                pos=pos_tensor,
                z=z_tensor,
                y_graph=energy_tensor,
                node_y=forces_tensor,
                edge_index=edge_index
            )
            
        except Exception as e:
            logger.debug(f"Error processing conformer {conf_idx}: {e}")
            return None

    def _get_edge_index(
        self, 
        positions: torch.Tensor, 
        cutoff: float = DEFAULT_CUTOFF
    ) -> torch.Tensor:
        """
        Create edge indices based on distance cutoff between atoms.
        
        This method generates graph connectivity by connecting atoms that are
        within a specified distance cutoff. If no edges are found (rare case),
        it falls back to connecting each atom to its nearest neighbor.
        
        Args:
            positions: Tensor of atomic coordinates [num_atoms, 3]
            cutoff: Distance cutoff for creating edges in Angstroms
            
        Returns:
            Edge index tensor [2, num_edges] defining graph connectivity
        """
        num_atoms = positions.shape[0]

        if num_atoms < 2:
            # Single atom - no edges
            return torch.empty((2, 0), dtype=torch.long)
        
        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(positions, positions)
        
        # Create edges for atoms within cutoff (excluding self-loops)
        mask = (dist_matrix < cutoff) & (dist_matrix > 0)
        edge_index = mask.nonzero().t()
        
        # Fallback: connect to nearest neighbors if no edges found
        if edge_index.shape[1] == 0:
            logger.debug(f"No edges found with cutoff {cutoff}, using nearest neighbors")
            edge_index = self._create_nearest_neighbor_edges(dist_matrix)
        
        return edge_index

    def _create_nearest_neighbor_edges(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Create edges by connecting each atom to its nearest neighbor.
        
        Args:
            dist_matrix: Distance matrix between atoms
            
        Returns:
            Edge index tensor with nearest neighbor connections
        """
        num_atoms = dist_matrix.shape[0]
        edge_list = []
        
        for i in range(num_atoms):
            distances = dist_matrix[i].clone()
            distances[i] = float('inf')  # Exclude self-connections
            
            if torch.any(torch.isfinite(distances)):
                nearest = torch.argmin(distances)
                # Add bidirectional edges
                edge_list.extend([[i, nearest.item()], [nearest.item(), i]])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            # Remove duplicate edges
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index

    def _save_processed_data(self, data_list: List[Data]) -> None:
        """
        Save processed data to disk.
        
        Args:
            data_list: List of processed Data objects
        """
        if not data_list:
            # Handle empty dataset
            logger.warning("No conformers processed - saving empty dataset")
            torch.save((None, None), self.processed_paths[0])
            return

        try:
            # Collate data and save
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            
            logger.info(
                f"Saved {len(data_list)} processed conformers to "
                f"{self.processed_paths[0]}"
            )
            logger.debug(f"Collated data structure: {data}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise 