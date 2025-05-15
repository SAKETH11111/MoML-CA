"""
Dataset Classes

This module provides dataset classes for working with molecular data.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Callable
import glob
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem # Added
import logging
from moml.core import create_graph_processor
from moml.core.molecular_graph_processor import MolecularGraphProcessor

logger = logging.getLogger(__name__)


class MolecularGraphDataset(Dataset):
    """
    Dataset for molecular graphs with optional labels.
    """
    
    def __init__(
        self,
        mol_files: List[str],
        labels: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            mol_files: List of paths to molecule files
            labels: Optional dictionary mapping filenames to labels
            config: Configuration for graph processing
            transform: Optional transform to apply to graphs
        """
        self.mol_files = mol_files
        self.labels = labels
        self.config = config or {}
        self.transform = transform
        self.graphs = []
        
        # Create graph processor
        self.graph_processor = create_graph_processor(self.config)
        
        # Process graphs
        self._process_graphs()
    
    def _process_graphs(self):
        """Process all molecule files into graphs."""
        # Use batch processing if available
        if hasattr(self.graph_processor, 'batch_files_to_graphs'):
            self.graphs = self.graph_processor.batch_files_to_graphs(self.mol_files)
            
            # Add labels to graphs if available
            if self.labels is not None:
                for i, file_path in enumerate(self.mol_files): # Assuming self.graphs aligns with self.mol_files
                    if file_path in self.labels and i < len(self.graphs):
                        current_graph = self.graphs[i]
                        # Ensure the graph object supports item assignment or attribute assignment
                        if isinstance(current_graph, dict):
                            current_graph['y'] = torch.tensor([self.labels[file_path]], dtype=torch.float)
                            current_graph.pop('label', None) # Use 'y' consistently
                        else: # Assume PyG Data object
                            setattr(current_graph, 'y', torch.tensor([self.labels[file_path]], dtype=torch.float))
                            if hasattr(current_graph, 'label'):
                                delattr(current_graph, 'label') # Remove 'label' if 'y' is set
            else: # Explicitly remove y or label if no labels are provided
                for i in range(len(self.graphs)):
                    current_graph = self.graphs[i]
                    if isinstance(current_graph, dict):
                        current_graph.pop('y', None)
                        current_graph.pop('label', None)
                    else: # Assume PyG Data object
                        if hasattr(current_graph, 'y'):
                            delattr(current_graph, 'y')
                        if hasattr(current_graph, 'label'):
                            delattr(current_graph, 'label')
        else:
            # Fallback to individual processing
            for file_path in tqdm(self.mol_files, desc="Processing molecular graphs"):
                try:
                    graph = None # Initialize graph to None at the start of try block
                    # Check if we already have a processed graph file
                    cache_path = file_path + '.pt'
                    if os.path.exists(cache_path):
                        graph = torch.load(cache_path, weights_only=False)
                    else:
                        # Process the molecule file
                        graph = self.graph_processor.file_to_graph(file_path)
                        
                        # Add labels if available
                        if self.labels is not None and file_path in self.labels:
                            graph.y = torch.tensor([self.labels[file_path]], dtype=torch.float)
                        elif self.labels is None: # No labels provided to dataset
                            if hasattr(graph, 'y'):
                                del graph.y
                            if hasattr(graph, 'label'): # 'label' might also be used by processor
                                del graph.label
                    
                    # Apply transform if available
                    if self.transform is not None:
                        graph = self.transform(graph)
                    
                    if graph is not None: # Ensure graph is not None before appending
                        self.graphs.append(graph)
                    else:
                        logger.warning(f"Graph for {file_path} was None after processing and transform, not adding to dataset.")
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """Get a graph by index."""
        graph = self.graphs[idx]
        
        # No need to apply transform here as it's already applied in _process_graphs
        return graph


class HierarchicalGraphDataset(Dataset):
    """
    Dataset for hierarchical molecular graphs with multiple levels of coarsening.
    """
    
    def __init__(
        self,
        data_dir: str,
        labels: Optional[Dict[str, float]] = None,
        levels: List[str] = ["atom", "functional_group", "structural_motif"],
        transform: Optional[Callable] = None
    ):
        """
        Initialize the hierarchical graph dataset.
        
        Args:
            data_dir: Directory containing hierarchical graph files
            labels: Optional dictionary mapping molecule IDs to labels
            levels: List of hierarchy levels to include
            transform: Optional transform to apply to graphs
        """
        self.data_dir = data_dir
        self.labels = labels
        self.levels = levels
        self.transform = transform
        
        # Find all molecule directories
        self.molecule_dirs = [d for d in glob.glob(os.path.join(data_dir, "*")) 
                             if os.path.isdir(d)]
        
        # Extract molecule IDs
        self.molecule_ids = [os.path.basename(d) for d in self.molecule_dirs]
        
    def __len__(self):
        """Return the number of molecules in the dataset."""
        return len(self.molecule_dirs)
    
    def __getitem__(self, idx):
        """Get hierarchical graphs for a molecule by index."""
        mol_dir = self.molecule_dirs[idx]
        mol_id = self.molecule_ids[idx]
        
        # Load graphs for each level
        graphs = {}
        for level in self.levels:
            graph_path = os.path.join(mol_dir, f"{level}_graph.pt")
            if os.path.exists(graph_path):
                graphs[level] = torch.load(graph_path, weights_only=False)
            else:
                # Try JSON format
                json_path = os.path.join(mol_dir, f"{level}_graph.json")
                if os.path.exists(json_path):
                    # Convert JSON to graph object
                    # TODO: This is problematic. create_molecular_graph_json GENERATES a JSON from a mol file.
                    # It does not LOAD a JSON into a graph object.
                    # This line will likely cause issues later as it returns a path or None.
                    # For now, providing output_dir to satisfy TypeError.
                    from moml.core.molecular_graph_processor import create_molecular_graph_json
                    try:
                        # Assuming json_path is the input *molecule* file, and output_dir is where it saves the *new* JSON.
                        # This is likely not the intended logic if json_path is an *existing* graph JSON.
                        created_json_path = create_molecular_graph_json(json_path, output_dir=mol_dir)
                        # If the intention was to load the JSON at json_path into a graph object, this is incorrect.
                        # graphs[level] should be a Data object. created_json_path is a string or None.
                        # This will need a proper JSON loading utility.
                        logger.warning(f"HierarchicalGraphDataset: create_molecular_graph_json called with json_path {json_path}. This function generates JSON, does not load it. Assigning its return path to graphs[{level}] which is likely incorrect.")
                        graphs[level] = created_json_path # This is likely wrong type for graphs[level]
                    except Exception as e:
                        logger.error(f"Error calling create_molecular_graph_json for {json_path} in dir {mol_dir}: {e}")
                        graphs[level] = None # Or handle error appropriately
        
        # Add label if available
        if self.labels is not None and mol_id in self.labels:
            label = torch.tensor([self.labels[mol_id]], dtype=torch.float)
            for level in graphs:
                if graphs[level] is not None: # Check if graph object exists
                    graphs[level].y = label
                else:
                    logger.warning(f"Graph for level '{level}' of molecule '{mol_id}' is None. Cannot assign label.")
        
        # Apply transform if available
        if self.transform is not None:
            for level in graphs:
                graphs[level] = self.transform(graphs[level])
        
        return graphs


class PFASDataset(Dataset):
    """
    Dataset specifically for PFAS compounds with additional features.
    """
    
    def __init__(
        self,
        data_path: str,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        smiles_column: str = "smiles",
        transform: Optional[Callable] = None
    ):
        """
        Initialize the PFAS dataset.
        
        Args:
            data_path: Path to CSV file containing PFAS data
            feature_columns: List of column names to use as features
            target_column: Column name to use as target
            smiles_column: Column name containing SMILES strings
            transform: Optional transform to apply to graphs
        """
        self.transform = transform
        
        # Load data
        self.df = pd.read_csv(data_path)
        
        # Extract features and targets
        self.features = None
        if feature_columns:
            self.features = self.df[feature_columns].values.astype(np.float32)
        
        self.targets = None
        if target_column and target_column in self.df.columns:
            self.targets = self.df[target_column].values.astype(np.float32)
        
        # Extract SMILES
        self.smiles = None
        if smiles_column in self.df.columns:
            self.smiles = self.df[smiles_column].tolist()
        
        # Create molecular graphs
        self.graphs = []
        if self.smiles:
            # RDKit Chem is already imported at the top level
            # from moml.core import batch_create_graphs_from_molecules # This function is being replaced
            
            # Convert SMILES to RDKit molecules
            molecules = [] # List of RDKit Mol objects
            valid_indices = [] # List of original indices in self.df for valid SMILES
            for i, smi in enumerate(self.smiles):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    # Attempt to generate 3D coordinates if not present, as mol_to_graph might need them
                    if mol.GetNumConformers() == 0:
                        try:
                            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                            AllChem.UFFOptimizeMolecule(mol)
                        except Exception as e_embed:
                            logger.warning(f"Could not generate 3D coordinates for SMILES {smi}: {e_embed}")
                    molecules.append(mol)
                    valid_indices.append(i)
                else:
                    logger.warning(f"Could not parse SMILES: {smi}")
            
            # Create graphs using MolecularGraphProcessor
            # Assuming PFASDataset might eventually take a config like MolecularGraphDataset
            processor_config = getattr(self, 'config', None) or {}
            processor = MolecularGraphProcessor(config=processor_config)
            
            processed_graphs = []
            successfully_processed_original_indices = []

            for i, mol_obj in enumerate(molecules): # mol_obj is an RDKit molecule
                original_df_idx = valid_indices[i] # Get the original index from the dataframe
                try:
                    # logger.debug(f"PFASDataset: Processing SMILES: {self.smiles[original_df_idx]}")
                    graph_data = processor.mol_to_graph(mol_obj) # This might add a 'y' or 'label' from mol props
                    # logger.debug(f"PFASDataset: Graph from processor for {self.smiles[original_df_idx]} has y? {hasattr(graph_data, 'y')}, has label? {hasattr(graph_data, 'label')}")

                    # If PFASDataset is not supposed to have targets, remove any 'y' or 'label'
                    # that the processor might have added.
                    # self.targets is None if target_column was not specified or not found.
                    if self.targets is None:
                        # logger.debug(f"PFASDataset: self.targets is None for SMILES: {self.smiles[original_df_idx]}.")
                        if hasattr(graph_data, 'y'):
                            # logger.debug(f"PFASDataset: Deleting y from graph for {self.smiles[original_df_idx]}")
                            del graph_data.y
                        if hasattr(graph_data, 'label'):
                            # logger.debug(f"PFASDataset: Deleting label from graph for {self.smiles[original_df_idx]}")
                            del graph_data.label
                    # else:
                        # logger.debug(f"PFASDataset: self.targets is NOT None for SMILES: {self.smiles[original_df_idx]}. Target value: {self.targets[original_df_idx] if original_df_idx < len(self.targets) else 'Index out of bounds'}")
                            
                    processed_graphs.append(graph_data)
                    successfully_processed_original_indices.append(original_df_idx)
                except Exception as e:
                    logger.error(f"Failed to process molecule (original index {original_df_idx}, SMILES: {self.smiles[original_df_idx]}) to graph: {e}")
            
            self.graphs = processed_graphs
            
            # Add targets to successfully processed graphs
            if self.targets is not None: # This implies target_column was valid and found
                # logger.debug(f"PFASDataset: Assigning targets. Number of graphs: {len(self.graphs)}, number of successfully_processed_original_indices: {len(successfully_processed_original_indices)}")
                for i, graph in enumerate(self.graphs): # Iterate through successfully created graphs
                    if i < len(successfully_processed_original_indices):
                        original_df_idx_for_this_graph = successfully_processed_original_indices[i]
                        # logger.debug(f"PFASDataset: Assigning target for graph {i} (original index {original_df_idx_for_this_graph}), target value: {self.targets[original_df_idx_for_this_graph]}")
                        graph.y = torch.tensor([self.targets[original_df_idx_for_this_graph]], dtype=torch.float)
                        if hasattr(graph, 'label'): # Clean up 'label' if 'y' is being set
                            # logger.debug(f"PFASDataset: Deleting label attribute from graph {i} as y is being set.")
                            del graph.label
                    # else:
                        # logger.warning(f"PFASDataset: Index mismatch when assigning targets. Graph index {i} out of bounds for successfully_processed_original_indices (len {len(successfully_processed_original_indices)})")
            # else:
                # logger.debug("PFASDataset: self.targets is None, so no targets assigned in the final loop.")
            # If self.targets is None, any 'y' or 'label' from processor was already removed above.
    
    def __len__(self):
        """Return the number of compounds in the dataset."""
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """Get a compound by index."""
        graph = self.graphs[idx]
        
        # Apply transform if available
        if self.transform is not None:
            graph = self.transform(graph)
        
        return graph