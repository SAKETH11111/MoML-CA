"""
moml/core/hierarchical_processor.py

Hierarchical graph processing for multi-scale molecular analysis.

This module implements hierarchical graph coarsening and multi-scale data processing
for HMGNN training. It generates multiple levels of graph representations from
atom-level to molecular fragment-level for comprehensive molecular analysis.

Main Components:
    - HierarchicalGraphCoarsener: Multi-level graph coarsening
    - HierarchicalDataProcessor: Data pipeline for hierarchical graphs
    - Scale mapping and cross-scale edge generation utilities
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx

# Optional dependencies with fallbacks
try:
    from torch_cluster import graclus_cluster
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False
    def graclus_cluster(*args, **kwargs):
        raise ImportError("torch_cluster is required for graclus clustering")

try:
    from torch_scatter import scatter_mean, scatter_add
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False
    def scatter_mean(src, index, dim=0, dim_size=None):
        """Fallback implementation of scatter_mean."""
        if src.numel() == 0:
            if dim_size is None:
                dim_size = 0
            return torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        
        if dim_size is None:
            dim_size = index.max().item() + 1 if index.numel() > 0 else 0
        
        # Handle different shapes
        result_shape = [dim_size] + list(src.shape[1:])
        result = torch.zeros(result_shape, dtype=src.dtype, device=src.device)
        
        for i in range(dim_size):
            mask = index == i
            if mask.any():
                result[i] = src[mask].mean(dim=0)
        return result
    
    def scatter_add(src, index, dim=0, dim_size=None):
        """Fallback implementation of scatter_add."""
        if src.numel() == 0:
            if dim_size is None:
                dim_size = 0
            return torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        
        if dim_size is None:
            dim_size = index.max().item() + 1 if index.numel() > 0 else 0
        
        # Handle different shapes
        result_shape = [dim_size] + list(src.shape[1:])
        result = torch.zeros(result_shape, dtype=src.dtype, device=src.device)
        
        for i in range(dim_size):
            mask = index == i
            if mask.any():
                result[i] = src[mask].sum(dim=0)
        return result

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Fragments
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

logger = logging.getLogger(__name__)


class HierarchicalGraphCoarsener:
    """
    Multi-level graph coarsening for hierarchical molecular representations.
    
    Generates hierarchical graph representations at different scales:
    - Level 0 (Fine): Atom-level graph
    - Level 1 (Medium): Functional group-level 
    - Level 2 (Coarse): Molecular fragment-level
    """
    
    def __init__(
        self,
        n_levels: int = 3,
        coarsening_ratios: Optional[List[float]] = None,
        clustering_method: str = "graclus",
        preserve_connectivity: bool = True,
        min_cluster_size: int = 2,
        max_cluster_size: int = 10
    ) -> None:
        """
        Initialize hierarchical graph coarsener.
        
        Args:
            n_levels: Number of hierarchical levels to generate
            coarsening_ratios: Ratio of nodes to keep at each level
            clustering_method: Method for clustering ('graclus', 'functional_groups', 'spectral')
            preserve_connectivity: Whether to preserve graph connectivity
            min_cluster_size: Minimum cluster size
            max_cluster_size: Maximum cluster size
        """
        self.n_levels = n_levels
        self.coarsening_ratios = coarsening_ratios or [1.0, 0.5, 0.25]
        self.clustering_method = clustering_method
        self.preserve_connectivity = preserve_connectivity
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        # Ensure we have enough coarsening ratios
        if len(self.coarsening_ratios) < n_levels:
            # Extend with geometric progression
            last_ratio = self.coarsening_ratios[-1]
            for i in range(len(self.coarsening_ratios), n_levels):
                last_ratio *= 0.5
                self.coarsening_ratios.append(last_ratio)
    
    def coarsen_graph(
        self, 
        data: Data, 
        level: int,
        mol: Optional[Any] = None
    ) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Coarsen a graph to the specified hierarchical level.
        
        Args:
            data: Input graph data
            level: Target coarsening level (0 = finest)
            mol: Optional RDKit molecule for functional group detection
            
        Returns:
            Tuple of (coarsened_data, node_mapping, cluster_counts)
        """
        if level == 0:
            # Level 0 is the original graph
            num_nodes = data.x.shape[0]
            identity_mapping = torch.arange(num_nodes, dtype=torch.long)
            cluster_counts = torch.ones(num_nodes, dtype=torch.long)
            return data, identity_mapping, cluster_counts
        
        # Determine clustering method based on level and availability
        if level == 1 and self.clustering_method == "functional_groups" and mol is not None and HAS_RDKIT:
            return self._functional_group_coarsening(data, mol)
        else:
            return self._geometric_coarsening(data, level)
    
    def _functional_group_coarsening(
        self, 
        data: Data, 
        mol: Any
    ) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Coarsen graph based on functional groups using RDKit.
        
        Args:
            data: Input graph data
            mol: RDKit molecule object
            
        Returns:
            Tuple of (coarsened_data, node_mapping, cluster_counts)
        """
        if not HAS_RDKIT:
            logger.warning("RDKit not available, falling back to geometric coarsening")
            return self._geometric_coarsening(data, 1)
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms != data.x.shape[0]:
            logger.warning(f"Atom count mismatch: mol={num_atoms}, data={data.x.shape[0]}")
            return self._geometric_coarsening(data, 1)
        
        # Detect functional groups
        functional_groups = self._detect_functional_groups(mol)
        
        # Create node mapping based on functional groups
        node_mapping = torch.arange(num_atoms, dtype=torch.long)
        cluster_id = 0
        
        # Assign atoms to functional group clusters
        assigned_atoms = set()
        for group_atoms in functional_groups:
            if len(group_atoms) >= self.min_cluster_size:
                for atom_idx in group_atoms:
                    if atom_idx not in assigned_atoms and atom_idx < num_atoms:
                        node_mapping[atom_idx] = cluster_id
                        assigned_atoms.add(atom_idx)
                cluster_id += 1
        
        # Assign unassigned atoms to individual clusters
        for atom_idx in range(num_atoms):
            if atom_idx not in assigned_atoms:
                node_mapping[atom_idx] = cluster_id
                cluster_id += 1
        
        # Count nodes per cluster
        num_clusters = cluster_id
        cluster_counts = torch.zeros(num_clusters, dtype=torch.long)
        for cluster in range(num_clusters):
            cluster_counts[cluster] = (node_mapping == cluster).sum()
        
        # Create coarsened graph
        coarsened_data = self._aggregate_clusters(data, node_mapping, num_clusters)
        
        return coarsened_data, node_mapping, cluster_counts
    
    def _geometric_coarsening(
        self, 
        data: Data, 
        level: int
    ) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Coarsen graph using geometric clustering methods.
        
        Args:
            data: Input graph data
            level: Coarsening level
            
        Returns:
            Tuple of (coarsened_data, node_mapping, cluster_counts)
        """
        target_ratio = self.coarsening_ratios[level]
        num_nodes = data.x.shape[0]
        target_clusters = max(1, int(num_nodes * target_ratio))
        
        if self.clustering_method == "graclus" and HAS_TORCH_CLUSTER:
            # Use GRACLUS clustering
            try:
                cluster_assignment = graclus_cluster(
                    data.edge_index, 
                    num_nodes=num_nodes,
                    weight=None
                )
                
                # Ensure we don't exceed target number of clusters
                unique_clusters = cluster_assignment.unique()
                if len(unique_clusters) > target_clusters:
                    # Merge smallest clusters
                    cluster_assignment = self._merge_small_clusters(
                        cluster_assignment, target_clusters
                    )
                
            except Exception as e:
                logger.warning(f"GRACLUS clustering failed: {e}, using random clustering")
                cluster_assignment = self._random_clustering(num_nodes, target_clusters)
        
        elif self.clustering_method == "graclus" and not HAS_TORCH_CLUSTER:
            logger.warning("torch_cluster not available, using random clustering instead of GRACLUS")
            cluster_assignment = self._random_clustering(num_nodes, target_clusters)
        
        else:
            # Fallback to random clustering
            cluster_assignment = self._random_clustering(num_nodes, target_clusters)
        
        # Count nodes per cluster
        num_clusters = cluster_assignment.max().item() + 1
        cluster_counts = torch.zeros(num_clusters, dtype=torch.long)
        for cluster in range(num_clusters):
            cluster_counts[cluster] = (cluster_assignment == cluster).sum()
        
        # Create coarsened graph
        coarsened_data = self._aggregate_clusters(data, cluster_assignment, num_clusters)
        
        return coarsened_data, cluster_assignment, cluster_counts
    
    def _detect_functional_groups(self, mol: Any) -> List[List[int]]:
        """
        Detect functional groups in a molecule using RDKit.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of lists containing atom indices for each functional group
        """
        functional_groups = []
        
        # Common functional group patterns
        patterns = {
            'carboxyl': '[CX3](=O)[OX2H1]',
            'hydroxyl': '[OX2H]',
            'amino': '[NX3;H2,H1;!$(NC=O)]',
            'carbonyl': '[CX3]=[OX1]',
            'ester': '[CX3](=O)[OX2H0]',
            'ether': '[OD2]([#6])[#6]',
            'amide': '[CX3](=[OX1])[NX3]',
            'sulfonic': '[SX4](=[OX1])(=[OX1])[OX2H1]',
            'phosphate': '[PX4](=[OX1])([OX2])([OX2])[OX2]',
            'halogen': '[F,Cl,Br,I]'
        }
        
        for group_name, pattern in patterns.items():
            try:
                patt = Chem.MolFromSmarts(pattern)
                if patt:
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        if len(match) >= self.min_cluster_size:
                            functional_groups.append(list(match))
            except Exception as e:
                logger.debug(f"Pattern matching failed for {group_name}: {e}")
        
        # Remove overlapping groups (keep larger ones)
        functional_groups = self._remove_overlapping_groups(functional_groups)
        
        return functional_groups
    
    def _remove_overlapping_groups(self, groups: List[List[int]]) -> List[List[int]]:
        """Remove overlapping functional groups, keeping larger ones."""
        groups = sorted(groups, key=len, reverse=True)
        non_overlapping = []
        used_atoms = set()
        
        for group in groups:
            if not any(atom in used_atoms for atom in group):
                non_overlapping.append(group)
                used_atoms.update(group)
        
        return non_overlapping
    
    def _random_clustering(self, num_nodes: int, target_clusters: int) -> torch.Tensor:
        """Create random clustering assignment."""
        cluster_assignment = torch.randint(0, target_clusters, (num_nodes,))
        return cluster_assignment
    
    def _merge_small_clusters(
        self, 
        cluster_assignment: torch.Tensor, 
        target_clusters: int
    ) -> torch.Tensor:
        """Merge small clusters to reach target number."""
        unique_clusters = cluster_assignment.unique()
        cluster_sizes = torch.bincount(cluster_assignment)
        
        # Sort clusters by size
        sorted_indices = torch.argsort(cluster_sizes)
        
        # Merge smallest clusters
        new_assignment = cluster_assignment.clone()
        clusters_to_merge = len(unique_clusters) - target_clusters
        
        for i in range(clusters_to_merge):
            small_cluster = sorted_indices[i].item()
            # Merge with the next larger cluster
            target_cluster = sorted_indices[clusters_to_merge].item()
            new_assignment[cluster_assignment == small_cluster] = target_cluster
        
        # Renumber clusters to be consecutive
        unique_new = new_assignment.unique()
        for i, cluster in enumerate(unique_new):
            new_assignment[new_assignment == cluster] = i
        
        return new_assignment
    
    def _aggregate_clusters(
        self, 
        data: Data, 
        cluster_assignment: torch.Tensor, 
        num_clusters: int
    ) -> Data:
        """
        Aggregate node features and create edges for clustered graph.
        
        Args:
            data: Original graph data
            cluster_assignment: Mapping from nodes to clusters
            num_clusters: Number of clusters
            
        Returns:
            Coarsened graph data
        """
        # Aggregate node features
        new_x = scatter_mean(data.x, cluster_assignment, dim=0, dim_size=num_clusters)
        
        # Create new edge indices
        row, col = data.edge_index
        new_row = cluster_assignment[row]
        new_col = cluster_assignment[col]
        
        # Remove self-loops and duplicate edges
        mask = new_row != new_col
        new_row = new_row[mask]
        new_col = new_col[mask]
        
        new_edge_index = torch.stack([new_row, new_col], dim=0)
        new_edge_index = torch.unique(new_edge_index, dim=1)
        
        # Aggregate edge attributes if present
        new_edge_attr = None
        if data.edge_attr is not None:
            # Create edge mapping
            edge_clusters = torch.stack([new_row, new_col], dim=0)
            unique_edges, inverse_indices = torch.unique(edge_clusters, dim=1, return_inverse=True)
            
            # Aggregate edge attributes
            new_edge_attr = scatter_mean(
                data.edge_attr[mask], 
                inverse_indices, 
                dim=0, 
                dim_size=unique_edges.shape[1]
            )
        
        # Create new data object
        coarsened_data = Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr
        )
        
        # Copy other attributes if present
        for key, value in data.__dict__.items():
            if key not in ['x', 'edge_index', 'edge_attr'] and not key.startswith('_'):
                if isinstance(value, torch.Tensor) and value.shape[0] == data.x.shape[0]:
                    # Aggregate node-level attributes
                    setattr(coarsened_data, key, scatter_mean(value, cluster_assignment, dim=0, dim_size=num_clusters))
                else:
                    # Copy as-is for graph-level attributes
                    setattr(coarsened_data, key, value)
        
        return coarsened_data
    
    def create_hierarchical_representation(
        self, 
        data: Data, 
        mol: Optional[Any] = None
    ) -> Tuple[List[Data], List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Create complete hierarchical representation of a graph.
        
        Args:
            data: Input graph data
            mol: Optional RDKit molecule for functional group detection
            
        Returns:
            Tuple of (scale_data_list, mappings_list, cluster_counts_list)
        """
        scale_data = []
        mappings = []
        cluster_counts = []
        
        current_data = data
        
        for level in range(self.n_levels):
            coarsened_data, mapping, counts = self.coarsen_graph(current_data, level, mol)
            
            scale_data.append(coarsened_data)
            mappings.append(mapping)
            cluster_counts.append(counts)
            
            # Use coarsened data for next level
            if level < self.n_levels - 1:
                current_data = coarsened_data
        
        cross_scale_edges = self._generate_cross_scale_edges(scale_data, mappings)
        return scale_data, mappings, cluster_counts, cross_scale_edges


class HierarchicalDataProcessor:
    """
    Data processor for creating hierarchical molecular datasets.
    
    Processes molecular data to create multi-scale representations suitable
    for HMGNN training, including cross-scale edge generation and batch processing.
    """
    
    def __init__(
        self,
        coarsener: Optional[HierarchicalGraphCoarsener] = None,
        include_cross_scale_edges: bool = True,
        cross_scale_edge_threshold: float = 5.0,
        cache_hierarchical: bool = True
    ) -> None:
        """
        Initialize hierarchical data processor.
        
        Args:
            coarsener: Graph coarsener instance
            include_cross_scale_edges: Whether to include cross-scale edges
            cross_scale_edge_threshold: Distance threshold for cross-scale edges
            cache_hierarchical: Whether to cache hierarchical representations
        """
        self.coarsener = coarsener or HierarchicalGraphCoarsener()
        self.include_cross_scale_edges = include_cross_scale_edges
        self.cross_scale_edge_threshold = cross_scale_edge_threshold
        self.cache_hierarchical = cache_hierarchical
        self._cache = {}
    
    def process_molecule(
        self, 
        data: Data, 
        mol: Optional[Any] = None,
        molecule_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single molecule to create hierarchical representation.
        
        Args:
            data: Input molecular graph data
            mol: Optional RDKit molecule object
            molecule_id: Optional identifier for caching
            
        Returns:
            Dictionary containing hierarchical data
        """
        # Check cache
        if self.cache_hierarchical and molecule_id and molecule_id in self._cache:
            return self._cache[molecule_id]
        
        # Create hierarchical representation
        scale_data, mappings, cluster_counts = self.coarsener.create_hierarchical_representation(data, mol)
        
        # Generate cross-scale edges if requested
        cross_scale_edges = None
        if self.include_cross_scale_edges:
            cross_scale_edges = self._generate_cross_scale_edges(scale_data, mappings)
        
        # Package results
        result = {
            'scale_data': scale_data,
            'mappings': mappings,
            'cluster_counts': cluster_counts,
            'cross_scale_edges': cross_scale_edges,
            'n_scales': len(scale_data)
        }
        
        # Cache if requested
        if self.cache_hierarchical and molecule_id:
            self._cache[molecule_id] = result
        
        return result
    
    def _generate_cross_scale_edges(
        self, 
        scale_data: List[Data], 
        mappings: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate cross-scale edges between adjacent hierarchical levels.
        
        Args:
            scale_data: List of data at different scales
            mappings: List of node mappings between scales
            
        Returns:
            List of (edge_index, edge_attr) tuples for cross-scale connections
        """
        cross_scale_edges = []
        
        for level in range(len(scale_data) - 1):
            fine_data = scale_data[level]
            coarse_data = scale_data[level + 1]
            mapping = mappings[level + 1]  # Maps fine to coarse
            
            # Create edges from fine nodes to their coarse representatives
            fine_nodes = torch.arange(fine_data.x.shape[0])
            coarse_nodes = mapping
            
            # Stack to create edge index [2, num_edges]
            edge_index = torch.stack([fine_nodes, coarse_nodes], dim=0)
            
            # Create simple edge attributes (can be enhanced)
            edge_attr = torch.ones(edge_index.shape[1], 1)
            
            cross_scale_edges.append((edge_index, edge_attr))
        
        return cross_scale_edges
    
    def create_batch(
        self, 
        hierarchical_data_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a batch from multiple hierarchical molecular representations.
        
        Args:
            hierarchical_data_list: List of hierarchical data dictionaries
            
        Returns:
            Batched hierarchical data
        """
        if not hierarchical_data_list:
            return {}
        
        n_scales = hierarchical_data_list[0]['n_scales']
        batch_size = len(hierarchical_data_list)
        
        # Initialize batch structures
        batched_scale_data = [[] for _ in range(n_scales)]
        batched_mappings = [[] for _ in range(n_scales)]
        batched_cluster_counts = [[] for _ in range(n_scales)]
        batched_cross_scale_edges = [[] for _ in range(n_scales - 1)]
        
        # Collect data from each molecule
        for mol_data in hierarchical_data_list:
            scale_data = mol_data['scale_data']
            mappings = mol_data['mappings']
            cluster_counts = mol_data['cluster_counts']
            cross_scale_edges = mol_data.get('cross_scale_edges', [])
            
            for scale in range(n_scales):
                batched_scale_data[scale].append(scale_data[scale])
                if scale < len(mappings):
                    batched_mappings[scale].append(mappings[scale])
                if scale < len(cluster_counts):
                    batched_cluster_counts[scale].append(cluster_counts[scale])
            
            for scale in range(n_scales - 1):
                if scale < len(cross_scale_edges):
                    batched_cross_scale_edges[scale].append(cross_scale_edges[scale])
        
        # Create PyG batches for each scale
        final_scale_data = []
        for scale in range(n_scales):
            if batched_scale_data[scale]:
                try:
                    batch_obj = Batch.from_data_list(batched_scale_data[scale])
                    final_scale_data.append({
                        'x': batch_obj.x,
                        'edge_index': batch_obj.edge_index,
                        'edge_attr': batch_obj.edge_attr,
                        'batch': batch_obj.batch
                    })
                except Exception as e:
                    logger.warning(f"Failed to create batch for scale {scale}: {e}")
                    # Create empty batch
                    final_scale_data.append({
                        'x': torch.empty(0, batched_scale_data[scale][0].x.shape[1]),
                        'edge_index': torch.empty(2, 0, dtype=torch.long),
                        'edge_attr': None,
                        'batch': torch.empty(0, dtype=torch.long)
                    })
        
        # Process mappings and cluster counts
        final_mappings = []
        final_cluster_counts = []
        
        for scale in range(n_scales):
            if batched_mappings[scale]:
                # Concatenate mappings with appropriate offsets
                offset = 0
                combined_mapping = []
                combined_counts = []
                
                for i, (mapping, counts) in enumerate(zip(batched_mappings[scale], batched_cluster_counts[scale])):
                    adjusted_mapping = mapping + offset
                    combined_mapping.append(adjusted_mapping)
                    combined_counts.append(counts)
                    offset += len(counts)
                
                final_mappings.append(torch.cat(combined_mapping))
                final_cluster_counts.append(torch.cat(combined_counts))
            else:
                final_mappings.append(torch.empty(0, dtype=torch.long))
                final_cluster_counts.append(torch.empty(0, dtype=torch.long))
        
        return {
            'scale_data': final_scale_data,
            'mappings': final_mappings,
            'cluster_counts': final_cluster_counts,
            'cross_scale_edges': batched_cross_scale_edges,
            'n_scales': n_scales,
            'batch_size': batch_size
        }


def create_hierarchical_processor(config: Dict[str, Any]) -> HierarchicalDataProcessor:
    """
    Factory function to create a hierarchical data processor.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured HierarchicalDataProcessor instance
    """
    coarsener_config = config.get('coarsener', {})
    processor_config = config.get('processor', {})
    
    coarsener = HierarchicalGraphCoarsener(**coarsener_config)
    processor = HierarchicalDataProcessor(coarsener=coarsener, **processor_config)
    
    return processor