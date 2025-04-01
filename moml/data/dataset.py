"""
Dataset Classes

This module provides dataset classes for working with molecular data.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import glob
from tqdm import tqdm 

from moml.core import create_graph_processor


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
                for i, file_path in enumerate(self.mol_files):
                    if file_path in self.labels:
                        self.graphs[i]['label'] = torch.tensor([self.labels[file_path]], dtype=torch.float)
        else:
            # Fallback to individual processing
            for file_path in tqdm(self.mol_files, desc="Processing molecular graphs"):
                try:
                    # Check if we already have a processed graph file
                    cache_path = file_path + '.pt'
                    if os.path.exists(cache_path):
                        graph = torch.load(cache_path)
                    else:
                        # Process the molecule file
                        graph = self.graph_processor.file_to_graph(file_path)
                        
                        # Add labels if available
                        if self.labels is not None and file_path in self.labels:
                            graph.y = torch.tensor([self.labels[file_path]], dtype=torch.float)
                    
                    # Apply transform if available
                    if self.transform is not None:
                        graph = self.transform(graph)
                    
                    self.graphs.append(graph)
                
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
                graphs[level] = torch.load(graph_path)
            else:
                # Try JSON format
                json_path = os.path.join(mol_dir, f"{level}_graph.json")
                if os.path.exists(json_path):
                    # Convert JSON to graph object
                    from moml.core import create_molecular_graph_json
                    graphs[level] = create_molecular_graph_json(json_path)
        
        # Add label if available
        if self.labels is not None and mol_id in self.labels:
            label = torch.tensor([self.labels[mol_id]], dtype=torch.float)
            for level in graphs:
                graphs[level].y = label
        
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
            from rdkit import Chem
            from moml.core import batch_create_graphs_from_molecules
            
            # Convert SMILES to RDKit molecules
            molecules = []
            valid_indices = []
            for i, smi in enumerate(self.smiles):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    molecules.append(mol)
                    valid_indices.append(i)
            
            # Create graphs
            self.graphs = batch_create_graphs_from_molecules(molecules)
            
            # Add targets to graphs
            if self.targets is not None:
                for i, idx in enumerate(valid_indices):
                    self.graphs[i].y = torch.tensor([self.targets[idx]], dtype=torch.float)
    
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