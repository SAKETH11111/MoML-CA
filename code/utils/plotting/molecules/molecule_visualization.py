#!/usr/bin/env python3
"""
Module for visualizing PFAS molecular structures.
"""

import os
import pickle
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def generate_2d_structure(mol, optimize_coords: bool = True) -> Chem.Mol:
    """
    Generate 2D coordinates for a molecule.
    
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
    mol, 
    size: Tuple[int, int] = (400, 300), 
    title: Optional[str] = None,
    alpha: float = 0.8
) -> np.ndarray:
    """
    Draw a single molecule as a PNG image.
    
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
    mol, 
    output_path: str, 
    size: Tuple[int, int] = (800, 600),
    title: Optional[str] = None
) -> str:
    """
    Save a molecule image to file.
    
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
    plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_molecule_grid(
    mols, 
    labels: Optional[List[str]] = None,
    size: Tuple[int, int] = (250, 200),
    n_cols: int = 4
) -> np.ndarray:
    """
    Generate a grid of molecule images.
    
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
    n_rows = (n_mols + n_cols - 1) // n_cols
    
    # Create a grid image
    img = Draw.MolsToGridImage(
        prepared_mols,
        molsPerRow=n_cols,
        subImgSize=size,
        legends=labels,
        useSVG=False
    )
    
    # Convert PIL image to numpy array
    return np.array(img)


def save_molecule_grid(
    mols,
    output_path: str,
    labels: Optional[List[str]] = None,
    size: Tuple[int, int] = (250, 200),
    n_cols: int = 4,
    title: Optional[str] = None
) -> str:
    """
    Save a grid of molecule images to file.
    
    Args:
        mols: List of RDKit molecule objects
        output_path: Path to save the image
        labels: Optional list of labels for each molecule
        size: Size of each molecule image (width, height)
        n_cols: Number of columns in the grid
        title: Optional title for the entire grid
        
    Returns:
        Path to the saved image
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate the grid
    grid_img = generate_molecule_grid(mols, labels, size, n_cols)
    
    # Save with matplotlib for better formatting
    fig = plt.figure(figsize=(grid_img.shape[1]/100, grid_img.shape[0]/100), dpi=100)
    
    if title:
        plt.title(title, fontsize=14)
    
    plt.imshow(grid_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return output_path


def visualize_dataset(
    mol_dict: Dict[str, Chem.Mol], 
    output_dir: str,
    grid_filename: str = "pfas20_compounds_grid.png",
    individual_dir: str = "individual",
    n_cols: int = 4
) -> Dict[str, str]:
    """
    Visualize a dataset of molecules, saving both a grid and individual images.
    
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
    save_molecule_grid(
        mols,
        grid_path,
        labels=names,
        n_cols=n_cols,
        title="PFAS Compounds"
    )
    output_paths["grid"] = grid_path
    
    # Save individual images
    individual_paths = {}
    for name, mol in mol_dict.items():
        img_path = os.path.join(individual_output_dir, f"{name}.png")
        save_molecule_image(mol, img_path, title=name)
        individual_paths[name] = img_path
    
    output_paths["individual"] = individual_paths
    
    return output_paths


def load_molecules_from_pickle(pickle_path: str) -> Dict[str, Chem.Mol]:
    """
    Load molecules from a pickle file.
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        Dictionary of molecules
    """
    with open(pickle_path, 'rb') as f:
        molecules = pickle.load(f)
    return molecules 