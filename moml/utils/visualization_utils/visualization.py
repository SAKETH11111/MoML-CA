#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consolidated Visualization Module

This module provides comprehensive visualization capabilities for:
1. Data analysis plots (distributions, correlations, etc.)
2. Molecular structure visualization
3. Molecular graph visualization
4. Dataset visualization

The module combines functionality from:
- Data visualization utilities
- Molecular structure visualization
- Graph visualization
- Dataset visualization
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from moml.core import MolecularGraphProcessor
from torch_geometric.data import Data as PyGData

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("visualization")

# ============================================================================
# Data Analysis Visualization
# ============================================================================


def plot_distribution(
    df: pd.DataFrame, column: str, xlabel: str, output_path: Union[str, Path], log_scale: bool = False, bins: int = 30
) -> None:
    """Plot distribution of a given column.

    Args:
        df: DataFrame containing the data
        column: Column name to plot
        xlabel: Label for x-axis
        output_path: Path to save the plot
        log_scale: Whether to use log scale
        bins: Number of bins for histogram
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), bins=bins, kde=True, log_scale=log_scale)
    plt.title(f"Distribution of {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.savefig(output_path)
    plt.close()


def plot_top_types(df: pd.DataFrame, column: str, title: str, output_path: Union[str, Path], top_n: int = 10) -> None:
    """Plot top types of a given column.

    Args:
        df: DataFrame containing the data
        column: Column name to plot
        title: Plot title
        output_path: Path to save the plot
        top_n: Number of top types to show
    """
    plt.figure(figsize=(12, 8))
    top_types = df[column].value_counts().head(top_n)
    sns.barplot(x=top_types.values, y=top_types.index)
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pie_chart(
    df: pd.DataFrame, column: str, labels: List[str], output_path: Union[str, Path], colors: Optional[List[str]] = None
) -> None:
    """Plot pie chart of a given column.

    Args:
        df: DataFrame containing the data
        column: Column name to plot
        labels: Labels for pie chart segments
        output_path: Path to save the plot
        colors: Colors for pie chart segments
    """
    plt.figure(figsize=(8, 6))
    counts = df[column].value_counts()
    if colors is None:
        colors = ["#ff9999", "#66b3ff"]
    plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors)
    plt.title(f"{column} Distribution")
    plt.savefig(output_path)
    plt.close()


def plot_heatmap(
    corr_matrix: pd.DataFrame, title: str, output_path: Union[str, Path], figsize: tuple = (12, 10)
) -> None:
    """Plot heatmap of correlation matrix.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Plot title
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    output_path: Union[str, Path],
    hue: Optional[str] = None,
    alpha: float = 0.6,
) -> None:
    """Plot scatter plot of two columns.

    Args:
        df: DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        output_path: Path to save the plot
        hue: Column name for color grouping
        alpha: Transparency of points
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=alpha)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.close()


def plot_success_rate(
    df: pd.DataFrame, group_column: str, success_column: str, output_path: Union[str, Path], top_n: int = 10
) -> None:
    """Plot success rate by group.

    Args:
        df: DataFrame containing the data
        group_column: Column to group by
        success_column: Column indicating success
        output_path: Path to save the plot
        top_n: Number of top groups to show
    """
    plt.figure(figsize=(12, 8))
    success_rate = df.groupby(group_column)[success_column].mean().sort_values(ascending=False).head(top_n)
    sns.barplot(x=success_rate.values * 100, y=success_rate.index)
    plt.title(f"Success Rate by {group_column} (Top {top_n})")
    plt.xlabel("Success Rate (%)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_count(df: pd.DataFrame, column: str, xlabel: str, output_path: Union[str, Path]) -> None:
    """Plot count of values in a column.

    Args:
        df: DataFrame containing the data
        column: Column name to plot
        xlabel: Label for x-axis
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column)
    plt.title(f"Count of {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.savefig(output_path)
    plt.close()


# ============================================================================
# Molecular Structure Visualization
# ============================================================================


def generate_2d_structure(mol: Chem.Mol, optimize_coords: bool = True) -> Chem.Mol:
    """Generate 2D coordinates for a molecule.

    Args:
        mol: RDKit molecule object
        optimize_coords: Whether to optimize the 2D coordinates

    Returns:
        RDKit molecule with 2D coordinates
    """
    mol_copy = Chem.Mol(mol)

    # Generate 2D coordinates if they don't exist
    if optimize_coords or not mol_copy.GetNumConformers():
        rdDepictor.Compute2DCoords(mol_copy)

    return mol_copy


def draw_molecule(
    mol: Chem.Mol, size: Tuple[int, int] = (400, 300), title: Optional[str] = None, alpha: float = 0.8
) -> np.ndarray:
    """Draw a single molecule as a PNG image.

    Args:
        mol: RDKit molecule object
        size: Image size (width, height) in pixels
        title: Optional title to display
        alpha: Opacity of the drawing

    Returns:
        NumPy array containing the image data
    """
    mol_with_coords = generate_2d_structure(mol)

    # Create a drawing object
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.SetFontSize(0.8)

    # Draw the molecule
    drawer.DrawMolecule(mol_with_coords, legend=title)
    drawer.FinishDrawing()

    # Convert to numpy array
    png_data = drawer.GetDrawingText()
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(png_data))
    return np.array(img)


def save_molecule_image(
    mol: Chem.Mol, output_path: str, size: Tuple[int, int] = (800, 600), title: Optional[str] = None
) -> str:
    """Save a molecule image to file.

    Args:
        mol: RDKit molecule object
        output_path: Path to save the image
        size: Image size (width, height) in pixels
        title: Optional title to display

    Returns:
        Path to the saved image
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Draw the molecule
    img = draw_molecule(mol, size, title)

    # Save using matplotlib to ensure proper formatting
    plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

    return output_path


def generate_molecule_grid(
    mols: List[Chem.Mol], labels: Optional[List[str]] = None, size: Tuple[int, int] = (250, 200), n_cols: int = 4
) -> np.ndarray:
    """Generate a grid of molecule images.

    Args:
        mols: List of RDKit molecule objects
        labels: Optional list of labels for each molecule
        size: Size of each molecule image (width, height)
        n_cols: Number of columns in the grid

    Returns:
        NumPy array containing the grid image
    """
    # Generate labels if needed
    if labels is None:
        labels = [None] * len(mols)

    # Prepare molecules with 2D coords
    prepared_mols = []
    for mol in mols:
        if mol is not None:
            prepared_mols.append(generate_2d_structure(mol))
        else:
            prepared_mols.append(None)

    # Calculate grid dimensions
    n_mols = len(prepared_mols)

    # Create a grid image
    img = Draw.MolsToGridImage(prepared_mols, molsPerRow=n_cols, subImgSize=size, legends=labels, useSVG=False)

    # Convert PIL image to numpy array
    return np.array(img)


def save_molecule_grid(
    mols: List[Chem.Mol],
    output_path: str,
    labels: Optional[List[str]] = None,
    size: Tuple[int, int] = (250, 200),
    n_cols: int = 4,
    title: Optional[str] = None,
) -> str:
    """Save a grid of molecular images to a file.

    Args:
        mols (List[Chem.Mol]): A list of RDKit molecule objects to display.
        output_path (str): The path to save the image file.
        labels (Optional[List[str]]): Optional list of labels for each molecule.
        size (Tuple[int, int]): The size (width, height) in pixels for each molecule image.
        n_cols (int): The number of columns in the grid.
        title (Optional[str]): An optional title for the entire grid.

    Returns:
        str: The path to the saved image file.
    """
    grid_img = generate_molecule_grid(mols, labels, size, n_cols)

    # Save with matplotlib for better formatting
    plt.figure(figsize=(grid_img.shape[1] / 100, grid_img.shape[0] / 100), dpi=100)

    if title:
        plt.title(title, fontsize=16)

    plt.imshow(grid_img)
    plt.axis("off")  # Remove axes

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    return output_path


def visualize_dataset(
    mol_dict: Dict[str, Chem.Mol],
    output_dir: str,
    grid_filename: str = "pfas20_compounds_grid.png",
    individual_dir: str = "individual",
    n_cols: int = 4,
) -> Dict[str, str]:
    """Visualize a dataset of molecules, saving both a grid and individual images.

    Args:
        mol_dict: Dictionary mapping compound names to RDKit molecules
        output_dir: Directory to save the visualizations
        grid_filename: Filename for the grid image
        individual_dir: Subdirectory for individual molecule images
        n_cols: Number of columns in the grid

    Returns:
        Dictionary with paths to the generated images
    """
    output_paths = {}

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    individual_output_dir = os.path.join(output_dir, individual_dir)
    os.makedirs(individual_output_dir, exist_ok=True)

    # Extract molecules and names
    names = list(mol_dict.keys())
    mols = [mol_dict[name] for name in names]

    # Save grid image
    grid_path = os.path.join(output_dir, grid_filename)
    save_molecule_grid(mols, grid_path, labels=names, n_cols=n_cols, title="PFAS Compounds")
    output_paths["grid"] = grid_path

    # Save individual images
    individual_paths = {}
    for name, mol in mol_dict.items():
        img_path = os.path.join(individual_output_dir, f"{name}.png")
        save_molecule_image(mol, img_path, title=name)
        individual_paths[name] = img_path

    output_paths["individual"] = individual_paths

    return output_paths


def load_molecules_from_pickle(pickle_file: str) -> Dict[str, Chem.Mol]:
    """Load RDKit molecules from a pickle file.

    Args:
        pickle_file: Path to the pickle file

    Returns:
        Dictionary mapping compound names to RDKit molecules
    """
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Molecule pickle file not found: {pickle_file}")

    try:
        with open(pickle_file, "rb") as f:
            molecules = pickle.load(f)

        # Ensure it's a dictionary
        if not isinstance(molecules, dict):
            raise ValueError(f"Expected a dictionary, got {type(molecules)}")

        # Verify that values are RDKit molecules
        for name, mol in molecules.items():
            if not isinstance(mol, Chem.Mol):
                raise ValueError(f"Entry '{name}' is not an RDKit molecule")

        return molecules

    except Exception as e:
        raise ValueError(f"Error loading molecules from pickle: {str(e)}")


# ============================================================================
# Molecular Graph Visualization
# ============================================================================


def visualize_molecular_graph(graph, output_file: Optional[str] = None, highlight_feature: str = "fluorine"):
    """Visualize a molecular graph with highlighted features.

    Args:
        graph: PyTorch Geometric Data object
        output_file: Path to save the visualization
        highlight_feature: Feature to highlight ('fluorine', 'partial_charge', 'functional_group', 'head_group')
    """
    try:
        import networkx as nx
        from torch_geometric.utils import to_networkx
    except ImportError:
        logger.error("Visualization requires networkx and PyTorch Geometric.")
        return

    # Check if the input is an RDKit Mol object and convert if necessary
    if isinstance(graph, Chem.Mol):
        processor = MolecularGraphProcessor()  # Assuming default config is okay for visualization
        pyg_graph = processor.mol_to_graph(graph)
        if pyg_graph is None:
            logger.error("Failed to convert RDKit Mol to PyG Data for visualization.")
            return
    elif isinstance(graph, PyGData):
        pyg_graph = graph
    else:
        logger.error(f"Unsupported graph type for visualization: {type(graph)}. Expected RDKit Mol or PyG Data.")
        return

    G = to_networkx(pyg_graph, to_undirected=True)

    plt.figure(figsize=(12, 10))

    # Use 3D positions for layout if available
    if hasattr(pyg_graph, "pos") and pyg_graph.pos is not None:
        pos = {i: (pyg_graph.pos[i][0].item(), pyg_graph.pos[i][1].item()) for i in range(pyg_graph.num_nodes)}
    else:
        pos = nx.spring_layout(G, seed=42)

    # Determine node colors based on highlight feature
    node_colors = []
    node_sizes = []

    # Determine atomic numbers for each node, preferring original RDKit mol if available
    atomic_nums = []
    original_rdkit_mol_available = isinstance(graph, Chem.Mol)

    for i in range(pyg_graph.num_nodes):
        if original_rdkit_mol_available:
            atomic_nums.append(graph.GetAtomWithIdx(i).GetAtomicNum())
        elif hasattr(pyg_graph, "x") and pyg_graph.x is not None and pyg_graph.x.shape[1] > 0:
            # Assuming the first feature in pyg_graph.x is atomic_number or a one-hot encoding of it.
            # This part is fragile and depends on MolecularGraphProcessor's feature generation.
            # For simplicity, let's assume if not RDKit Mol, we can't reliably get atomic_num for highlighting here
            # without knowing the exact feature vector structure.
            # A more robust way would be to ensure 'atomic_num' is stored in pyg_graph nodes if not RDKit mol.
            # For now, let's try to get it from G.nodes if populated by to_networkx, else default to 0.
            node_data = G.nodes[i]
            if "atomic_num" in node_data:  # if to_networkx added it
                atomic_nums.append(node_data["atomic_num"])
            elif pyg_graph.x[i][0].item() < 20:  # Crude check if first element is atomic_num like
                atomic_nums.append(int(pyg_graph.x[i][0].item()))
            else:  # Fallback
                atomic_nums.append(0)  # Unknown
        else:
            atomic_nums.append(0)  # Unknown

    for i in range(pyg_graph.num_nodes):  # Use pyg_graph here
        size = 500  # Default node size
        atom_idx = i  # current atom index

        is_f_atom = False
        is_c_bonded_to_f = False

        if highlight_feature == "fluorine":
            current_atomic_num = atomic_nums[atom_idx]
            is_f_atom = current_atomic_num == 9

            if current_atomic_num == 6:  # If it's a Carbon atom
                if original_rdkit_mol_available:
                    rdkit_atom = graph.GetAtomWithIdx(atom_idx)
                    for neighbor in rdkit_atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 9:
                            is_c_bonded_to_f = True
                            break
                # Else (if only PyG graph), determining C-F bonds from pyg_graph.x is complex
                # without knowing specific feature indices for C-F bonds or neighbor types.
                # For now, is_c_bonded_to_f will remain False if not an RDKit Mol.

            if is_f_atom:
                node_colors.append("red")
                size = 700
            elif is_c_bonded_to_f:
                node_colors.append("lightcoral")
                size = 600
            else:
                node_colors.append("lightblue")  # Other atoms

        elif highlight_feature == "partial_charge":
            # Placeholder: This section requires partial charge data to be present in pyg_graph.x
            # or accessible from the original RDKit molecule if it has charges.
            # For now, it will default to lightblue.
            # This part needs actual partial charge data.
            # For now, it will default to lightblue.
            node_colors.append("lightblue")  # Default color for this placeholder

        elif highlight_feature == "functional_group":
            # Placeholder: This requires functional group information.
            # If original RDKit mol is available, FunctionalGroupDetector could be used.
            # If PyG graph, specific features for COOH, SO3H etc. would be needed.
            node_colors.append("lightblue")  # Default color for this placeholder

        elif highlight_feature == "head_group":
            # Placeholder: This requires head group vs tail information.
            node_colors.append("lightblue")  # Default color for this placeholder

        else:  # Default coloring if no specific highlight_feature matches or for non-PFAS highlights
            current_atomic_num = atomic_nums[atom_idx]
            if current_atomic_num == 9:  # Fluorine
                node_colors.append("red")
                size = 700
            elif current_atomic_num == 6:  # Carbon
                is_c_bonded_to_f_default = False
                if original_rdkit_mol_available:
                    rdkit_atom = graph.GetAtomWithIdx(atom_idx)
                    for neighbor in rdkit_atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 9:
                            is_c_bonded_to_f_default = True
                            break
                if is_c_bonded_to_f_default:
                    node_colors.append("lightcoral")
                else:
                    node_colors.append("gray")
            elif current_atomic_num == 8:  # Oxygen
                node_colors.append("skyblue")
            else:
                node_colors.append("lightgrey")  # Default for other atoms

        node_sizes.append(size)

    # Draw the graph
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        width=2.0,
        with_labels=True,
        font_size=10,
        font_weight="bold",
    )

    # Add title based on highlight feature
    plt.title(f'Molecular Graph - {highlight_feature.replace("_", " ").title()} Highlight', fontsize=16)

    # Save or display the visualization
    # fig was defined earlier by fig = plt.figure(...)
    # Explicitly get the current figure to ensure 'fig_to_return' is correctly assigned.
    fig_to_return = plt.gcf()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        logger.info(f"Graph visualization saved to {output_file}")
        plt.close(fig_to_return)  # Close the specific figure instance after saving
        return fig_to_return
    else:
        # If not saving to file, return the figure object.
        # The caller is responsible for plt.show() or plt.close(fig_to_return).
        return fig_to_return


def print_graph_statistics(graph):
    """Print statistics about a molecular graph.

    Args:
        graph: PyTorch Geometric Data object
    """
    logger.info("\nGraph Statistics:")
    logger.info(f"  Number of nodes (atoms): {graph.num_nodes}")
    logger.info(
        f"  Number of edges (bonds): {graph.edge_index.shape[1] // 2}"
    )  # Divide by 2 because graph is undirected
    logger.info(f"  Node feature dimension: {graph.x.shape[1]}")
    logger.info(f"  Edge feature dimension: {graph.edge_attr.shape[1]}")

    # Count atom types if features follow the expected format
    if graph.x.shape[1] >= 10:
        fluorine_count = sum(1 for i in range(graph.num_nodes) if graph.x[i][8].item() > 0.5)
        carbon_fluorine_count = sum(1 for i in range(graph.num_nodes) if graph.x[i][9].item() > 0.5)
        logger.info(f"  Fluorine atoms: {fluorine_count}")
        logger.info(f"  Carbon atoms bonded to fluorine: {carbon_fluorine_count}")

    # Print global features
    if hasattr(graph, "y") and graph.y is not None:
        logger.info("\nGlobal Molecular Features:")
        logger.info(f"  Molecular weight: {graph.y[0].item():.2f}")
        logger.info(f"  Topological polar surface area: {graph.y[1].item():.2f}")
        logger.info(f"  H-bond donors: {graph.y[2].item():.0f}")
        logger.info(f"  H-bond acceptors: {graph.y[3].item():.0f}")
        logger.info(f"  LogP (octanol-water partition coefficient): {graph.y[4].item():.2f}")
        logger.info(f"  Total atoms: {graph.y[5].item():.0f}")
        logger.info(f"  Fluorine atoms: {graph.y[6].item():.0f}")

        # PFAS-specific global features (if available)
        if len(graph.y) > 7:
            logger.info(f"  CF3 groups: {graph.y[7].item():.0f}")
        if len(graph.y) > 10:
            logger.info(f"  Carboxylic acid groups: {graph.y[8].item():.0f}")
            logger.info(f"  Sulfonic acid groups: {graph.y[9].item():.0f}")
            logger.info(f"  Phosphonic acid groups: {graph.y[10].item():.0f}")
        if len(graph.y) > 13:
            logger.info(f"  CF groups: {graph.y[11].item():.0f}")
            logger.info(f"  CF2 groups: {graph.y[12].item():.0f}")
            logger.info(f"  CF3 groups: {graph.y[13].item():.0f}")


def visualize_hierarchical_graphs(hierarchical_graphs: Dict[str, str], output_dir: Optional[str] = None):
    """Visualize hierarchical graphs at each level.

    Args:
        hierarchical_graphs: Dictionary mapping level names to graph file paths
        output_dir: Directory to save visualizations
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for level, graph_path in hierarchical_graphs.items():
        try:
            graph = torch.load(graph_path)

            if output_dir:
                base_name = os.path.splitext(os.path.basename(graph_path))[0]
                viz_path = os.path.join(output_dir, f"{base_name}_{level}_visualization.png")
            else:
                viz_path = None

            visualize_molecular_graph(graph, viz_path, highlight_feature="functional_group")
            logger.info(f"Visualized {level} graph from {graph_path}")

            # Print statistics for this level
            logger.info(f"\n{level.replace('_', ' ').title()} Graph:")
            print_graph_statistics(graph)

        except Exception as e:
            logger.error(f"Error visualizing {level} graph: {e}")


# ============================================================================
# Main Function for PFAS-20 Visualization
# ============================================================================


def visualize_pfas20_molecules(
    input_path: str = "data/processed/chemical_list/pfas20_rdkit_mols.pkl",
    output_dir: str = "data/processed/visualizations",
    grid_filename: str = "pfas20_compounds_grid.png",
    n_cols: int = 4,
) -> Dict[str, str]:
    """Visualize the standardized PFAS-20 molecular structures.

    Args:
        input_path: Path to pickle file with processed molecules
        output_dir: Directory to save visualizations
        grid_filename: Filename for the molecule grid image
        n_cols: Number of columns in the grid

    Returns:
        Dictionary with paths to the generated images
    """
    logger.info(f"Loading PFAS-20 molecules from: {input_path}")
    molecules = load_molecules_from_pickle(input_path)
    logger.info(f"Loaded {len(molecules)} molecules")

    logger.info(f"Generating PFAS-20 visualizations in: {output_dir}")
    visualization_paths = visualize_dataset(molecules, output_dir, grid_filename=grid_filename, n_cols=n_cols)

    # Print summary
    logger.info("\nVisualization Summary:")
    logger.info(f"- Grid image: {visualization_paths['grid']}")
    logger.info(
        f"- Individual images: {len(visualization_paths['individual'])} files in {os.path.dirname(list(visualization_paths['individual'].values())[0])}"
    )

    return visualization_paths


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Visualize PFAS-20 molecules")
    parser.add_argument(
        "--input",
        "-i",
        default="data/processed/chemical_list/pfas20_rdkit_mols.pkl",
        help="Path to pickle file with processed molecules",
    )
    parser.add_argument(
        "--output-dir", "-o", default="data/processed/visualizations", help="Directory to save visualizations"
    )
    parser.add_argument(
        "--grid-filename", "-g", default="pfas20_compounds_grid.png", help="Filename for the molecule grid image"
    )
    parser.add_argument("--columns", "-c", type=int, default=4, help="Number of columns in the molecule grid")
    args = parser.parse_args()

    visualize_pfas20_molecules(
        input_path=args.input, output_dir=args.output_dir, grid_filename=args.grid_filename, n_cols=args.columns
    )
