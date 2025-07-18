"""
moml/core/hierarchical_graph_coarsener.py

Graph coarsening for PFAS molecular structures using hierarchical representations.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from rdkit import Chem
from moml.core.molecular_graph_processor import Data

from moml.core.molecular_graph_processor import MolecularGraphProcessor
from moml.core.molecular_feature_extraction import (
    FunctionalGroupDetector,
    MolecularFeatureExtractor,
)
from moml.simulation.qm.parser.orca_parser import parse_orca_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphCoarsener:
    """
    Creates hierarchical graph representations of PFAS molecules.

    Graph coarsening theory:
    The coarsening process aggregates atoms into meaningful chemical groups, creating a
    multi-resolution representation of the molecular structure. This follows principles from
    algebraic multigrid theory, where:

    1. Restriction: Fine-grained information is mapped to coarser representations
    2. Prolongation: Coarse-grained information is mapped back to fine-grained representations
    3. Clustering: Similar nodes are grouped based on chemical meaning

    Benefits:
    - Reduced computational complexity: O(n) → O(n/c) where c is the coarsening factor
    - Enhanced receptive field: Captures long-range interactions without deep architectures
    - Chemical interpretability: Graph nodes represent meaningful chemical functional groups
    - Multi-scale learning: Different chemical properties are evident at different scales

    The coarsening is performed in a chemically-guided manner rather than using generic
    graph algorithms, ensuring that the resulting super-nodes have chemical meaning
    (e.g., CF3 groups, carboxylic acid groups) specific to PFAS structures.
    """
    UNMAPPED_MOTIF = -1  # Identifier for unclassified structural motifs

    def __init__(self, use_3d_coords: bool = True, use_pfas_features: bool = True) -> None:
        """
        Initialize the GraphCoarsener.

        Args:
            use_3d_coords: Whether to include 3D coordinates in coarsened graphs
            use_pfas_features: Whether to include PFAS-specific features
        """
        self.use_3d_coords = use_3d_coords
        self.use_pfas_features = use_pfas_features
        self.functional_group_detector = FunctionalGroupDetector()
        self.feature_extractor = MolecularFeatureExtractor()

        # Create a graph processor for initial atom-level graphs
        self.graph_processor = MolecularGraphProcessor({
            "use_3d_coords": use_3d_coords, 
            "use_pfas_specific_features": use_pfas_features
        })

    def _create_cluster_mapping(self, mol: Chem.Mol) -> Dict[int, int]:
        """
        Create a mapping from atom indices to cluster indices for functional group level.

        Args:
            mol: RDKit molecule

        Returns:
            Dictionary mapping atom indices to cluster indices
        """
        # Identify functional groups
        cf_group_definitions, non_cf_functional_groups = (
            self.functional_group_detector.identify_all_functional_groups(mol)
        )

        logger.info(f"Molecule: {Chem.MolToSmiles(mol)}")
        logger.info(
            f"Identified non_cf_functional_groups type: {type(non_cf_functional_groups)}"
        )
        logger.info(
            f"Identified non_cf_functional_groups content: {non_cf_functional_groups}"
        )

        # Create initial mapping with each atom as its own cluster
        cluster_mapping = {i: i for i in range(mol.GetNumAtoms())}
        next_cluster_id = mol.GetNumAtoms()

        # Assign cluster IDs to non-CF functional groups
        logger.debug(
            f"Full non_cf_functional_groups before loop: {non_cf_functional_groups}"
        )
        for i, group_atom_indices in enumerate(non_cf_functional_groups):
            logger.debug(
                f"Loop iteration {i}, group_atom_indices type: "
                f"{type(group_atom_indices)}, content: {group_atom_indices}"
            )
            logger.info(
                f"Processing group {i}, type: {type(group_atom_indices)}, "
                f"content: {group_atom_indices}"
            )
            
            if isinstance(group_atom_indices, (set, list, tuple)):
                # This is expected: group_atom_indices is a set of atom indices
                current_group_cluster_id = next_cluster_id
                for atom_idx in group_atom_indices:
                    if isinstance(atom_idx, int):  # Ensure atom_idx is an integer
                        cluster_mapping[atom_idx] = current_group_cluster_id
                    else:
                        logger.warning(
                            f"Unexpected item {atom_idx} (type {type(atom_idx)}) "
                            f"in functional group set {group_atom_indices}. Skipping."
                        )
                next_cluster_id += 1
            else:
                err_msg = (
                    f"Unexpected type for a functional group element: "
                    f"{type(group_atom_indices)}. Content: {group_atom_indices}. "
                    f"Expected set, list, or tuple of atom indices."
                )
                logger.error(err_msg)
                raise TypeError(err_msg)

        # CF groups (CF, CF2, CF3) are typically handled at the atom feature level
        # or as individual nodes if they are not part of a larger clustered functional group.
        # The current clustering logic aims to group multiple atoms (like COOH) into one supernode.
        # For CFx, the carbon atom itself is the "group".
        # If CFx groups (e.g., the C of a CF3) should also form their own distinct clusters
        # separate from other atoms, that logic would need to be added here.
        # However, the original code's comment "excluding CF groups" and "Handle CF groups - they're already identified at the atom level"
        # suggests they are not clustered in this step.
        # The cf_group_definitions (Dict[int, str]) maps C atom index to 'CF', 'CF2', 'CF3'.
        # We are not using cf_group_definitions to create new clusters here, aligning with the idea
        # that these are atom-level properties or single-atom "groups" for coarsening purposes.

        return cluster_mapping

    def _create_structural_mapping(
        self, mol: Chem.Mol, cluster_mapping: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Create a mapping from cluster indices to structural motif indices.
        
        This method identifies major structural components of PFAS molecules:
        - Head groups (functional groups like COOH, SO3H)
        - Tail groups (fluorinated carbon chains)

        Args:
            mol: RDKit molecule
            cluster_mapping: Mapping from atom indices to functional group cluster indices

        Returns:
            Dictionary mapping cluster indices to structural motif indices:
            - 0: Head group motif (functional groups)
            - 1: Tail group motif (fluorinated chains)
        """
        distance_features = self.feature_extractor.calculate_distance_features(mol)

        # Initialize structural mapping
        structural_mapping = {}

        # Define motif identifiers
        HEAD_GROUP_MOTIF = 0  # Functional groups (COOH, SO3H, etc.)
        TAIL_GROUP_MOTIF = 1  # Fluorinated carbon chains

        for atom_idx, cluster_id in cluster_mapping.items():
            # Skip if this cluster is already mapped
            if cluster_id in structural_mapping:
                continue

            # Check if atom is in head group based on distance features
            is_head = False
            if atom_idx in distance_features:
                is_head = distance_features[atom_idx]["is_head_group"] > 0.5

            # Assign cluster to head or tail structural motif
            if is_head:
                structural_mapping[cluster_id] = HEAD_GROUP_MOTIF
            else:
                structural_mapping[cluster_id] = TAIL_GROUP_MOTIF

        return structural_mapping

    def _compute_coarsened_features(self, data: Data, cluster_mapping: Dict[int, int]) -> torch.Tensor:
        """
        Compute node features for the coarsened graph by aggregating original features.

        Args:
            data: Original PyTorch Geometric Data object
            cluster_mapping: Mapping from atom indices to cluster indices

        Returns:
            Tensor of node features for the coarsened graph
        """
        # Get original node features
        x = data.x
        if x is None:
            raise ValueError("Input data.x (node features) is None.")

        # Create a reverse mapping from cluster IDs to atom indices
        clusters = {}
        for atom_idx, cluster_id in cluster_mapping.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(atom_idx)

        # Calculate features for each cluster by averaging the features of its atoms
        coarsened_features = []
        for cluster_id in sorted(clusters.keys()):
            atom_indices = clusters[cluster_id]
            # Average the features of all atoms in this cluster
            cluster_features = torch.mean(x[atom_indices], dim=0)
            coarsened_features.append(cluster_features)

        return torch.stack(coarsened_features)

    def _compute_coarsened_edges(
        self, data: Data, cluster_mapping: Dict[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edges for the coarsened graph.

        Args:
            data: Original PyTorch Geometric Data object
            cluster_mapping: Mapping from atom indices to cluster indices

        Returns:
            Tuple of (edge_index, edge_attr) for the coarsened graph
        """
        # Get original edges
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if edge_index is None or edge_attr is None:
            raise ValueError("Input data.edge_index or data.edge_attr is None.")

        # Create a set of edges between clusters
        cluster_edges = set()
        cluster_edge_attrs = {}

        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i].item())
            dst = int(edge_index[1, i].item())

            # Safely map to cluster IDs, skip if missing
            src_cluster = cluster_mapping.get(src)
            dst_cluster = cluster_mapping.get(dst)
            if src_cluster is None or dst_cluster is None:
                continue

            # Skip self-loops within the same cluster
            if src_cluster == dst_cluster:
                continue

            # Add edge between clusters
            edge = (src_cluster, dst_cluster)
            cluster_edges.add(edge)

            # Aggregate edge attributes
            if edge not in cluster_edge_attrs:
                cluster_edge_attrs[edge] = []
            cluster_edge_attrs[edge].append(edge_attr[i])

        # Create edge index and attribute tensors
        coarsened_edge_index = []
        coarsened_edge_attr = []

        for edge in cluster_edges:
            src_cluster, dst_cluster = edge
            coarsened_edge_index.append([src_cluster, dst_cluster])

            # Average edge attributes
            avg_attr = torch.mean(torch.stack(cluster_edge_attrs[edge]), dim=0)
            coarsened_edge_attr.append(avg_attr)

        if not coarsened_edge_index:
            # No edges in the coarsened graph
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                (0, edge_attr.shape[1] if edge_attr is not None else 0), dtype=torch.float
            )

        return (torch.tensor(coarsened_edge_index, dtype=torch.long).t(), torch.stack(coarsened_edge_attr))

    def _compute_coarsened_positions(self, data: Data, cluster_mapping: Dict[int, int]) -> Optional[torch.Tensor]:
        """
        Compute 3D positions for the coarsened graph by averaging positions of atoms in each cluster.

        Args:
            data: Original PyTorch Geometric Data object
            cluster_mapping: Mapping from atom indices to cluster indices

        Returns:
            Tensor of 3D positions for the coarsened graph, or None if no positions are available
        """
        if not hasattr(data, "pos") or data.pos is None:
            return None

        pos = data.pos
        # Create a reverse mapping from cluster IDs to atom indices
        clusters = {}
        for atom_idx, cluster_id in cluster_mapping.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(atom_idx)

        # Calculate positions for each cluster by averaging the positions of its atoms
        coarsened_positions = []
        for cluster_id in sorted(clusters.keys()):
            atom_indices = clusters[cluster_id]
            # Average the positions of all atoms in this cluster
            cluster_pos = torch.mean(pos[atom_indices], dim=0)
            coarsened_positions.append(cluster_pos)

        return torch.stack(coarsened_positions)

    def create_functional_group_graph(self, data: Data, mol: Chem.Mol) -> Data:
        """
        Create a coarsened graph at the functional group level.

        Args:
            data: Original PyTorch Geometric Data object
            mol: RDKit molecule

        Returns:
            PyTorch Geometric Data object for the coarsened graph
        """
        # Create mapping from atoms to functional group clusters
        cluster_mapping = self._create_cluster_mapping(mol)

        # Compute features, edges, and positions for the coarsened graph
        coarsened_x = self._compute_coarsened_features(data, cluster_mapping)
        coarsened_edge_index, coarsened_edge_attr = self._compute_coarsened_edges(data, cluster_mapping)
        coarsened_pos = self._compute_coarsened_positions(data, cluster_mapping) if self.use_3d_coords else None

        # Create the coarsened graph
        coarsened_data = Data(
            x=coarsened_x,
            edge_index=coarsened_edge_index,
            edge_attr=coarsened_edge_attr,
            pos=coarsened_pos,
            y=data.y,  # Keep the same global features
            num_nodes=coarsened_x.shape[0],
            # Store the cluster mapping for reference
            cluster_mapping=cluster_mapping,
        )

        # Transfer any custom attributes from the original graph
        keys_to_iterate_fg = []
        if hasattr(data, "keys"):
            if callable(data.keys):  # For dict-like objects
                keys_to_iterate_fg = list(data.keys())
            else:  # For PyG Data objects where data.keys is a list/property
                keys_to_iterate_fg = data.keys

        for key in keys_to_iterate_fg:
            if key not in ["x", "edge_index", "edge_attr", "pos", "y", "num_nodes"]:
                if hasattr(data, key):  # Ensure the key actually exists on data
                    coarsened_data[key] = data[key]

        return coarsened_data

    def create_structural_motif_graph(self, atom_level_data: Data, mol: Chem.Mol) -> Data:
        """
        Create a coarsened graph at the structural motif level from an atom-level graph.

        Args:
            atom_level_data: Original atom-level PyTorch Geometric Data object.
            mol: RDKit molecule corresponding to atom_level_data.

        Returns:
            PyTorch Geometric Data object for the structural motif level graph.
        """
        # 1. Create Functional Group (FG) graph from the atom-level graph.
        # This step also computes the mapping from original atom indices to FG cluster IDs.
        functional_group_graph = self.create_functional_group_graph(atom_level_data, mol)

        # This mapping is from original atom indices to the nodes (FGs) of the functional_group_graph.
        # It's stored as 'cluster_mapping' on the functional_group_graph.
        atom_to_fg_mapping = functional_group_graph.cluster_mapping
        if not isinstance(atom_to_fg_mapping, dict):
            raise TypeError(f"Internal error: atom_to_fg_mapping should be a dict, got {type(atom_to_fg_mapping)}")

        # 2. Create mapping from FG cluster IDs to structural motif IDs.
        # _create_structural_mapping uses the RDKit molecule and the atom_to_fg_mapping
        # to determine head/tail motifs for each FG cluster.
        # It returns a dict: {fg_cluster_id: motif_id}
        fg_to_motif_mapping = self._create_structural_mapping(mol, atom_to_fg_mapping)
        if not isinstance(fg_to_motif_mapping, dict):
            raise TypeError(f"Internal error: fg_to_motif_mapping should be a dict, got {type(fg_to_motif_mapping)}")

        # 3. Create a combined mapping from original atom indices directly to structural motif IDs.
        atom_to_motif_mapping = {}
        for atom_idx, fg_cluster_id in atom_to_fg_mapping.items():
            motif_id = fg_to_motif_mapping.get(fg_cluster_id, self.UNMAPPED_MOTIF)
            if motif_id == self.UNMAPPED_MOTIF:
                logger.warning(
                    f"FG cluster {fg_cluster_id} (from atom {atom_idx}) was not mapped to a known motif. "
                    f"Assigning default ID {self.UNMAPPED_MOTIF}."
                )
            atom_to_motif_mapping[atom_idx] = motif_id

        # Unmapped FG clusters are assigned a default ID to ensure all atoms are included in the motif graph.
        
        # 4. Compute features, edges, and positions for the structural motif graph.
        # These computations MUST use the original atom_level_data and the atom_to_motif_mapping.
        motif_x = self._compute_coarsened_features(atom_level_data, atom_to_motif_mapping)
        motif_edge_index, motif_edge_attr = self._compute_coarsened_edges(atom_level_data, atom_to_motif_mapping)
        motif_pos = (
            self._compute_coarsened_positions(atom_level_data, atom_to_motif_mapping) if self.use_3d_coords else None
        )

        # Create the structural motif graph Data object
        motif_data = Data(
            x=motif_x,
            edge_index=motif_edge_index,
            edge_attr=motif_edge_attr,
            pos=motif_pos,
            y=atom_level_data.y,  # Global features from the original atom-level graph
            num_nodes=motif_x.shape[0] if motif_x is not None and motif_x.dim() > 0 else 0,  # Handle empty motif_x
            # Store relevant mappings for potential downstream use or debugging
            atom_to_fg_mapping=atom_to_fg_mapping,
            fg_to_motif_mapping=fg_to_motif_mapping,
            atom_to_motif_mapping=atom_to_motif_mapping,  # This is the direct atom-to-motif map
        )

        # Transfer any other custom attributes from the original atom-level graph
        keys_to_iterate_motif = []
        if hasattr(atom_level_data, "keys"):
            if callable(atom_level_data.keys):  # For dict-like objects
                keys_to_iterate_motif = list(atom_level_data.keys())
            else:  # For PyG Data objects where data.keys is a list/property
                keys_to_iterate_motif = atom_level_data.keys

        # Prepare an iterable for motif_data's keys as well for the 'in' check
        motif_data_keys_iterable = []
        if hasattr(motif_data, "keys"):
            if callable(motif_data.keys):
                motif_data_keys_iterable = list(motif_data.keys())
            else:
                motif_data_keys_iterable = motif_data.keys

        for key in keys_to_iterate_motif:
            if key not in motif_data_keys_iterable:  # Use the prepared iterable
                if hasattr(atom_level_data, key):  # Ensure the key actually exists on atom_level_data
                    motif_data[key] = atom_level_data[key]

        return motif_data

    def create_hierarchical_graphs(
        self, data: Data, mol: Chem.Mol
    ) -> Dict[str, Data]:
        """
        Create hierarchical graph representations at multiple levels of coarseness.
        
        This is the main method for generating multi-scale molecular representations.
        It creates three levels of graph abstractions:
        1. Atom level - Original molecular graph with individual atoms
        2. Functional group level - Atoms grouped into chemical functional groups
        3. Structural motif level - Functional groups grouped into major structural components

        Args:
            data: Original PyTorch Geometric Data object (atom-level)
            mol: RDKit molecule corresponding to the data

        Returns:
            Dictionary of graphs at different levels:
            - 'atom': Original atom-level graph
            - 'functional_group': Functional group level graph
            - 'structural_motif': Structural motif level graph
        """
        # Create functional group level graph
        functional_group_graph = self.create_functional_group_graph(data, mol)

        # Create structural motif level graph  
        structural_motif_graph = self.create_structural_motif_graph(
            functional_group_graph, mol  # Use functional group graph
        )

        return {
            "atom": data, 
            "functional_group": functional_group_graph, 
            "structural_motif": structural_motif_graph
        }

    def create_from_molecule_file(
        self,
        mol_file: str,
        output_dir: Optional[str] = None,
        use_partial_charges: bool = False,
        partial_charges: Optional[List[float]] = None,
    ) -> Dict[str, Union[str, Data]]:
        """
        Create hierarchical graph representations from a molecule file.

        Args:
            mol_file: Path to the molecule file (MOL/SDF)
            output_dir: Directory to save graph files (default: same directory as mol_file)
            use_partial_charges: Whether to include partial charges in the graph
            partial_charges: Optional list of partial charges

        Returns:
            Dictionary mapping level names to paths of saved graph files or Data objects
        """
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.dirname(mol_file)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load molecule
        mol = Chem.MolFromMolFile(mol_file, removeHs=False)
        if mol is None:
            raise ValueError(f"Failed to load molecule from {mol_file}")

        # Create base atom-level graph
        base_name = os.path.splitext(os.path.basename(mol_file))[0]

        # Create graph processor with updated configuration
        processor_config = {
            "use_pfas_specific_features": self.use_pfas_features,
            "use_3d_coords": self.use_3d_coords,
            "use_partial_charges": use_partial_charges,
        }

        graph_processor = MolecularGraphProcessor(processor_config)

        # Process the molecule file with partial charges if available
        if use_partial_charges and partial_charges is not None:
            atom_graph_data = graph_processor.file_to_graph(
                mol_file, additional_features={"partial_charges": partial_charges}
            )
        else:
            atom_graph_data = graph_processor.file_to_graph(mol_file)

        if atom_graph_data is None:
            raise ValueError(f"Failed to create atom-level graph from {mol_file}")

        # Generate hierarchical graphs
        hierarchical_graphs = self.create_hierarchical_graphs(
            cast(Data, atom_graph_data), mol
        )

        # Save graphs if output_dir is provided, otherwise return graph objects
        if output_dir:
            graph_paths = {}

            # Save atom-level graph
            atom_graph_path = os.path.join(output_dir, f"{base_name}_atom_graph.pt")
            torch.save(hierarchical_graphs["atom"], atom_graph_path)
            graph_paths["atom"] = atom_graph_path

            # Save other levels
            for level, graph in hierarchical_graphs.items():
                if level == "atom":
                    continue  # Already saved

                # Save this level
                graph_path = os.path.join(output_dir, f"{base_name}_{level}_graph.pt")
                torch.save(graph, graph_path)
                graph_paths[level] = graph_path

            return graph_paths
        else:
            # Cast to correct type for return
            return dict(hierarchical_graphs)

    def create_from_orca(
        self,
        mol_file: str,
        orca_output: str,
        output_dir: Optional[str] = None,
        charge_type: str = "mulliken",
        use_quantum_properties: bool = True,
    ) -> Dict[str, Union[str, Data]]:
        """
        Create hierarchical graph representations from a molecule file and ORCA output.

        Args:
            mol_file: Path to the molecule file (MOL/SDF)
            orca_output: Path to the ORCA output file
            output_dir: Directory to save graph files (default: same directory as mol_file)
            charge_type: Type of partial charges to use ('mulliken' or 'loewdin')
            use_quantum_properties: Whether to include quantum properties from ORCA

        Returns:
            Dictionary mapping level names to paths of saved graph files or Data objects
        """
        # Parse ORCA output for quantum data
        qm_data = None
        if use_quantum_properties and os.path.exists(orca_output):
            qm_data = parse_orca_output(orca_output)

        # Extract partial charges if quantum properties are available
        partial_charges = None
        if use_quantum_properties and qm_data is not None:
            # Only use .get if qm_data is a dict
            if isinstance(qm_data, dict):
                charge_dict = qm_data.get(charge_type, None)
                if isinstance(charge_dict, dict):
                    partial_charges = charge_dict.get("charges", None)
                else:
                    partial_charges = None
            elif isinstance(qm_data, (list, tuple)):
                partial_charges = qm_data
            elif isinstance(qm_data, np.ndarray):
                partial_charges = qm_data.tolist()
            # else: leave as None

        # Create hierarchical graphs using the molecule file
        return self.create_from_molecule_file(
            mol_file=mol_file,
            output_dir=output_dir,
            use_partial_charges=use_quantum_properties and partial_charges is not None,
            partial_charges=partial_charges,
        )

    def batch_create_from_directories(
        self,
        mol_dir: str,
        orca_dir: str,
        output_dir: str,
        charge_type: str = "mulliken",
        use_quantum_properties: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """
        Batch process multiple molecules to create hierarchical graph representations.

        Args:
            mol_dir: Directory containing molecule files
            orca_dir: Directory containing ORCA output files
            output_dir: Directory to save graph files
            charge_type: Type of partial charges to use ('mulliken' or 'loewdin')
            use_quantum_properties: Whether to include quantum properties from ORCA

        Returns:
            Dictionary mapping molecule names to dictionaries of graph paths
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_graph_paths = {}

        # Find all molecule files
        mol_files = {}
        for filename in os.listdir(mol_dir):
            if filename.endswith(".mol") or filename.endswith(".sdf"):
                base_name = os.path.splitext(filename)[0]
                mol_files[base_name] = os.path.join(mol_dir, filename)

        # Find all ORCA output files
        orca_files = {}
        for filename in os.listdir(orca_dir):
            if filename.endswith(".out") or filename.endswith(".log"):
                base_name = os.path.splitext(filename)[0]
                orca_files[base_name] = os.path.join(orca_dir, filename)

        # Process each molecule with matching ORCA output
        for base_name, mol_file in mol_files.items():
            mol_output_dir = os.path.join(output_dir, base_name)

            # Find matching ORCA output
            orca_file = None
            if base_name in orca_files:
                orca_file = orca_files[base_name]
            else:
                # Look for similar names
                for orca_name, orca_path in orca_files.items():
                    if base_name in orca_name or orca_name in base_name:
                        orca_file = orca_path
                        break

            if orca_file is None and use_quantum_properties:
                print(f"No matching ORCA output found for {base_name}, processing without quantum data")

            try:
                # Create hierarchical graphs
                if orca_file and use_quantum_properties:
                    graph_paths = self.create_from_orca(
                        mol_file=mol_file,
                        orca_output=orca_file,
                        output_dir=mol_output_dir,
                        charge_type=charge_type,
                        use_quantum_properties=use_quantum_properties,
                    )
                else:
                    graph_paths = self.create_from_molecule_file(
                        mol_file=mol_file, output_dir=mol_output_dir, use_partial_charges=False
                    )

                all_graph_paths[base_name] = graph_paths
                print(f"Created hierarchical graphs for {base_name}")

            except Exception as e:
                print(f"Error creating hierarchical graphs for {base_name}")

        return all_graph_paths
