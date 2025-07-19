"""
moml/data/pfas_sdf_dataset.py

PFAS (Per- and Polyfluoroalkyl Substances) dataset from SDF files.

This module provides a PyTorch Geometric dataset class for loading and processing
PFAS molecules from SDF (Structure Data Format) files. It converts molecular
structures into graph representations with computed molecular descriptors as
targets, making it suitable for machine learning tasks on PFAS compounds.

Classes:
    PFASSDFDataset: Dataset class for PFAS molecules with molecular descriptors

Example:
    Basic usage for loading PFAS dataset:
    
    >>> from moml.data.pfas_sdf_dataset import PFASSDFDataset
    >>> dataset = PFASSDFDataset(root='data/pfas_sdf/')
    >>> print(f"Dataset size: {len(dataset)}")
    >>> graph = dataset[0]
    >>> print(f"Node features: {graph.z.shape}")
    >>> print(f"Molecular descriptors: {graph.y.shape}")
"""

import glob
import logging
import os
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

# Constants
NUM_DESCRIPTORS = 19
DEFAULT_DESCRIPTOR_VALUE = 0.0

# Configure module logger
logger = logging.getLogger(__name__)

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Crippen, Descriptors, rdDepictor  # type: ignore
    from rdkit import RDLogger  # type: ignore
    HAS_RDKIT = True

    # Suppress RDKit warnings for cleaner output
    RDLogger.DisableLog('rdApp.*')  # type: ignore
    
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available. PFAS SDF dataset functionality will be limited.")
    
    # Create dummy classes for fallback
    class Chem:  # type: ignore
        """Dummy Chem class when RDKit is not available."""
        
        class Mol:
            """Dummy Mol class."""
            pass
        
        @staticmethod
        def SDMolSupplier(sdf_path, removeHs=False, sanitize=True):
            """Dummy supplier that returns empty list."""
            return []
    
    class Descriptors:  # type: ignore
        """Dummy Descriptors class."""
        
        @staticmethod
        def MolWt(mol): return 0.0
        
        @staticmethod
        def ExactMolWt(mol): return 0.0
        
        @staticmethod
        def TPSA(mol): return 0.0
        
        @staticmethod
        def NumHAcceptors(mol): return 0
        
        @staticmethod
        def NumHDonors(mol): return 0
        
        @staticmethod
        def NumRotatableBonds(mol): return 0
        
        @staticmethod
        def NumAromaticRings(mol): return 0
        
        @staticmethod
        def NumSaturatedRings(mol): return 0
        
        @staticmethod
        def BalabanJ(mol): return 0.0
        
        @staticmethod
        def BertzCT(mol): return 0.0
        
        @staticmethod
        def HallKierAlpha(mol): return 0.0
        
        @staticmethod
        def Kappa1(mol): return 0.0
        
        @staticmethod
        def Kappa2(mol): return 0.0
        
        @staticmethod
        def Kappa3(mol): return 0.0
        
        @staticmethod
        def LabuteASA(mol): return 0.0
        
        @staticmethod
        def NumHeteroatoms(mol): return 0
    
    class Crippen:  # type: ignore
        """Dummy Crippen class."""
        
        @staticmethod
        def MolLogP(mol): return 0.0
    
    class rdDepictor:  # type: ignore
        """Dummy rdDepictor class."""
        
        @staticmethod
        def Compute2DCoords(mol): pass


class PFASSDFDataset(InMemoryDataset):
    """
    Dataset for PFAS molecules loaded from SDF files.
    
    This dataset class loads PFAS (Per- and Polyfluoroalkyl Substances) molecules
    from SDF (Structure Data Format) files and converts them into PyTorch Geometric
    graph representations. Each molecule is processed to extract atomic features,
    3D coordinates, bond connectivity, and computed molecular descriptors.
    
    The dataset computes 19-dimensional molecular descriptors as graph-level
    targets, making it compatible with QM9-style machine learning tasks while
    focusing specifically on PFAS compounds.
    
    Args:
        root: Root directory containing 'raw' and 'processed' subdirectories
        split: Optional dataset split identifier (e.g., 'train', 'test')
        transform: Optional transform to apply to each graph during loading
        pre_transform: Optional transform to apply during processing
        pre_filter: Optional filter to apply during processing
    
    Attributes:
        split: Dataset split identifier
        data: Processed graph data
        slices: Data slicing information for batching
    
    Example:
        >>> dataset = PFASSDFDataset(root='data/pfas_molecules/')
        >>> print(f"Number of molecules: {len(dataset)}")
        >>> graph = dataset[0]
        >>> print(f"Atomic numbers: {graph.z}")
        >>> print(f"3D coordinates: {graph.pos}")
        >>> print(f"Molecular descriptors: {graph.y}")
    
    Note:
        SDF files should be placed in the 'raw' subdirectory of the root path.
        Processed data will be saved to the 'processed' subdirectory.
    
    Raises:
        ImportError: If RDKit is not available
        FileNotFoundError: If no SDF files are found in the raw directory
    """
    
    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        """
        Initialize the PFAS SDF dataset.
        
        Args:
            root: Root directory path
            split: Optional split identifier
            transform: Optional transform function
            pre_transform: Optional pre-processing transform
            pre_filter: Optional pre-processing filter
            
        Raises:
            ImportError: If RDKit is not available
        """
        if not HAS_RDKIT:
            raise ImportError("RDKit is required for PFASSDFDataset")
        
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        
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
                loaded_data = torch.load(processed_file)
                
                # Check if data is valid (not None)
                if loaded_data[0] is not None:
                    self.data, self.slices = loaded_data
                    logger.info(f"Loaded processed PFAS data from {processed_file}")
                else:
                    self.data, self.slices = None, None
                    logger.warning("Processed data file contains None values")
                    
            except Exception as e:
                logger.error(f"Failed to load processed data from {processed_file}: {e}")
                self.data, self.slices = None, None
        else:
            self.data, self.slices = None, None
            logger.debug("No processed data file found")
        
    @property
    def raw_file_names(self) -> List[str]:
        """
        Return list of SDF files in the raw directory.
        
        Returns:
            List of SDF filenames (basenames only, not full paths)
        """
        try:
            sdf_pattern = os.path.join(self.raw_dir, "*.sdf")
            sdf_files = glob.glob(sdf_pattern)
            filenames = [os.path.basename(f) for f in sdf_files]
            
            if not filenames:
                logger.warning(f"No SDF files found in {self.raw_dir}")
            else:
                logger.debug(f"Found {len(filenames)} SDF files")
                
            return filenames
            
        except Exception as e:
            logger.error(f"Error finding SDF files in {self.raw_dir}: {e}")
            return []
    
    @property
    def processed_file_names(self) -> List[str]:
        """
        Return processed file names based on split.
        
        Returns:
            List containing the processed file name
        """
        split_suffix = f"_{self.split}" if self.split else ""
        return [f"pfas_sdf{split_suffix}.pt"]
    
    def download(self) -> None:
        """
        Check for presence of SDF files (no actual download).
        
        This method validates that SDF files are present in the raw directory.
        Since PFAS SDF files are typically provided locally, no actual
        downloading is performed.
        
        Raises:
            FileNotFoundError: If no SDF files are found in raw directory
        """
        if not self.raw_file_names:
            error_msg = f"No SDF files found in {self.raw_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Found {len(self.raw_file_names)} SDF files in raw directory")
    
    def process(self) -> None:
        """
        Process SDF files into PyTorch Geometric Data objects.
        
        This method reads all SDF files in the raw directory, converts each
        molecule to a graph representation with molecular descriptors, and
        saves the processed data for future use.
        
        The processing includes:
        - Reading molecules from SDF files
        - Converting to graph format (nodes, edges, positions)
        - Computing molecular descriptors as targets
        - Applying pre-filtering and pre-transforms if specified
        - Saving processed data to disk
        """
        logger.info("Starting PFAS SDF processing")
        data_list = []
        total_files = len(self.raw_file_names)
        successful_molecules = 0
        failed_files = 0
        
        for i, sdf_file in enumerate(self.raw_file_names):
            logger.debug(f"Processing file {i+1}/{total_files}: {sdf_file}")
            sdf_path = os.path.join(self.raw_dir, sdf_file)
            
            try:
                # Read molecules from SDF file
                molecules = self._read_sdf_file(sdf_path)
                
                for mol in molecules:
                    if mol is None:
                        continue
                        
                    # Convert molecule to Data object
                    data = self._mol_to_data(mol)
                    if data is not None:
                        data_list.append(data)
                        successful_molecules += 1
                        
            except Exception as e:
                logger.error(f"Error processing {sdf_file}: {e}")
                failed_files += 1
                continue
        
        # Log processing summary
        logger.info(
            f"Processing complete: {successful_molecules} molecules from "
            f"{total_files - failed_files}/{total_files} files"
        )
        
        # Apply pre-processing filters and transforms
        data_list = self._apply_preprocessing(data_list)
        
        # Save processed data
        self._save_processed_data(data_list)

    def _read_sdf_file(self, sdf_path: str) -> List:
        """
        Read molecules from an SDF file.
        
        Args:
            sdf_path: Path to SDF file
            
        Returns:
            List of RDKit molecule objects
        """
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
            return list(suppl)
        except Exception as e:
            logger.error(f"Failed to read SDF file {sdf_path}: {e}")
            return []

    def _apply_preprocessing(self, data_list: List[Data]) -> List[Data]:
        """
        Apply pre-filtering and pre-transforms to data list.
        
        Args:
            data_list: List of Data objects
            
        Returns:
            Processed list of Data objects
        """
        initial_count = len(data_list)
        
        # Apply pre_filter if specified
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            filtered_count = len(data_list)
            if filtered_count < initial_count:
                logger.info(
                    f"Pre-filter removed {initial_count - filtered_count} molecules"
                )
            
        # Apply pre_transform if specified
        if self.pre_transform is not None:
            try:
                data_list = [self.pre_transform(data) for data in data_list]
                logger.info("Applied pre-transform to all molecules")
            except Exception as e:
                logger.error(f"Error applying pre-transform: {e}")
        
        return data_list

    def _save_processed_data(self, data_list: List[Data]) -> None:
        """
        Save processed data to disk.
        
        Args:
            data_list: List of processed Data objects
        """
        if not data_list:
            # Handle empty dataset
            logger.warning("No valid molecules to save - saving empty dataset")
            torch.save((None, None), self.processed_paths[0])
            return

        try:
            # Collate data and save
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])  # type: ignore[arg-type]
            logger.info(
                f"Saved {len(data_list)} processed molecules to "
                f"{self.processed_paths[0]}"
            )
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise

    def _mol_to_data(self, mol) -> Optional[Data]:
        """
        Convert RDKit molecule to PyTorch Geometric Data object.
        
        This method extracts atomic features, 3D coordinates, bond connectivity,
        and molecular descriptors from an RDKit molecule and packages them into
        a PyTorch Geometric Data object suitable for graph neural networks.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            PyTorch Geometric Data object or None if conversion fails
        """
        try:
            # Get atomic information
            atoms = mol.GetAtoms()
            num_atoms = len(atoms)
            
            if num_atoms == 0:
                logger.debug("Molecule has no atoms, skipping")
                return None
            
            # Extract atomic features
            node_features = self._extract_node_features(atoms)
            
            # Extract 3D coordinates
            positions = self._extract_positions(mol, num_atoms)
            
            # Extract bond connectivity
            edge_index = self._extract_edges(mol)
            
            # Compute molecular descriptors as targets
            descriptors = self._compute_molecular_descriptors(mol)
            
            return Data(
                z=node_features,
                pos=positions,
                edge_index=edge_index,
                y=descriptors
            )
            
        except Exception as e:
            logger.debug(f"Error converting molecule to data: {e}")
            return None

    def _extract_node_features(self, atoms) -> torch.Tensor:
        """
        Extract atomic numbers as node features.
        
        Args:
            atoms: List of RDKit atom objects
            
        Returns:
            Tensor of atomic numbers
        """
        atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
        return torch.tensor(atomic_numbers, dtype=torch.long)

    def _extract_positions(self, mol, num_atoms: int) -> torch.Tensor:
        """
        Extract 3D atomic coordinates.
        
        Args:
            mol: RDKit molecule object
            num_atoms: Number of atoms in molecule
            
        Returns:
            Tensor of 3D coordinates [num_atoms, 3]
        """
        try:
            # Try to get existing 3D conformer
            conf = mol.GetConformer()
            positions = []
            
            for i in range(num_atoms):
                atom_pos = conf.GetAtomPosition(i)
                positions.append([atom_pos.x, atom_pos.y, atom_pos.z])
                
            return torch.tensor(positions, dtype=torch.float)
            
        except (AttributeError, ValueError):
            # If no 3D conformer, generate 2D coordinates and set z=0
            logger.debug("No 3D conformer found, using 2D coordinates")
        
        try:
            rdDepictor.Compute2DCoords(mol)
            conf = mol.GetConformer()
            positions = []
            
            for i in range(num_atoms):
                atom_pos = conf.GetAtomPosition(i)
                positions.append([atom_pos.x, atom_pos.y, 0.0])
            
            return torch.tensor(positions, dtype=torch.float)
            
        except Exception as e:
            logger.warning(f"Failed to generate coordinates: {e}")
            # Return zero coordinates as fallback
            return torch.zeros((num_atoms, 3), dtype=torch.float)

    def _extract_edges(self, mol) -> torch.Tensor:
        """
        Extract bond connectivity as edge indices.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Tensor of edge indices [2, num_edges]
        """
        edge_indices = []
    
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Add both directions for undirected graph
            edge_indices.extend([[i, j], [j, i]])
        
        if edge_indices:
            return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            # No bonds - return empty edge_index
            return torch.empty((2, 0), dtype=torch.long)
            
    def _compute_molecular_descriptors(self, mol) -> torch.Tensor:
        """
        Compute molecular descriptors as graph-level targets.
        
        This method computes a comprehensive set of molecular descriptors
        that characterize the chemical and physical properties of PFAS
        molecules. The descriptors are designed to be compatible with
        QM9-style target dimensions while being relevant for PFAS compounds.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Tensor of molecular descriptors [19]
            
        Note:
            The 19 descriptors include basic molecular properties,
            topological indices, and PFAS-specific features like
            fluorine atom count.
        """
        try:
            # Compute comprehensive molecular descriptors
            descriptors = [
                Descriptors.MolWt(mol),                    # Molecular weight
                Descriptors.ExactMolWt(mol),               # Exact molecular weight  
                Crippen.MolLogP(mol),                      # LogP (lipophilicity)
                Descriptors.TPSA(mol),                     # Topological polar surface area
                Descriptors.NumHAcceptors(mol),            # H-bond acceptors
                Descriptors.NumHDonors(mol),               # H-bond donors
                Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
                Descriptors.NumAromaticRings(mol),         # Aromatic rings
                Descriptors.NumSaturatedRings(mol),        # Saturated rings
                mol.GetNumHeavyAtoms(),                    # Heavy atom count
                Descriptors.BalabanJ(mol),                 # Balaban J index
                Descriptors.BertzCT(mol),                  # Bertz complexity
                Descriptors.HallKierAlpha(mol),            # Hall-Kier alpha
                Descriptors.Kappa1(mol),                   # Kappa shape index 1
                Descriptors.Kappa2(mol),                   # Kappa shape index 2
                Descriptors.Kappa3(mol),                   # Kappa shape index 3
                Descriptors.LabuteASA(mol),                # Labute accessible surface area
                Descriptors.NumHeteroatoms(mol),           # Heteroatom count
                self._count_fluorine_atoms(mol),           # Fluorine count (PFAS-specific)
            ]
            
            # Validate and clean descriptor values
            clean_descriptors = []
            for _, desc in enumerate(descriptors):
                try:
                    value = float(desc)
                    if np.isnan(value) or np.isinf(value):
                        value = DEFAULT_DESCRIPTOR_VALUE
                    clean_descriptors.append(value)
                except (TypeError, ValueError):
                    clean_descriptors.append(DEFAULT_DESCRIPTOR_VALUE)
            
            return torch.tensor(clean_descriptors, dtype=torch.float)
            
        except Exception as e:
            logger.warning(f"Error computing molecular descriptors: {e}")
            # Return zeros if computation fails
            return torch.zeros(NUM_DESCRIPTORS, dtype=torch.float)

    def _count_fluorine_atoms(self, mol) -> int:
        """
        Count fluorine atoms in the molecule.
        
        This is a PFAS-specific descriptor since PFAS compounds are
        characterized by their fluorine content.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Number of fluorine atoms
        """
        try:
            return len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F'])
        except Exception:
            return 0
    
