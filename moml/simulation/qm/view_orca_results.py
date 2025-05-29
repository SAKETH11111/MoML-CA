#!/usr/bin/env python3
"""
Utility script for visualizing and summarizing ORCA computation results.
This script reads the ML training data file and provides a summary of the calculations.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("view_orca_results")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Utility for viewing ORCA computation results")

    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing ORCA results")

    parser.add_argument("--output_dir", type=str, default="visualization", help="Directory for saving visualizations")

    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["all", "homo_lumo", "charges", "energies", "structures"],
        default="all",
        help="Type of visualization to generate",
    )

    return parser.parse_args()


def load_ml_data(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load ML training data from results directory.

    Args:
        results_dir: Directory containing results

    Returns:
        List of molecule data dictionaries
    """
    ml_data_file = os.path.join(results_dir, "ml_training_data.json")

    if not os.path.exists(ml_data_file):
        logger.error(f"ML training data file not found: {ml_data_file}")
        return []

    try:
        with open(ml_data_file, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded data for {len(data)} molecules")
        return data

    except Exception as e:
        logger.error(f"Error loading ML data: {str(e)}")
        return []


def create_summary_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary DataFrame from ML data.

    Args:
        data: List of molecule data dictionaries

    Returns:
        DataFrame with summary information
    """
    summary = []

    for mol_data in data:
        summary.append(
            {
                "molecule_id": mol_data["molecule_id"],
                "smiles": mol_data["smiles"],
                "total_energy": mol_data["total_energy"],
                "homo": mol_data["homo"],
                "lumo": mol_data["lumo"],
                "gap": mol_data["gap"],
                "num_atoms": len(mol_data["geometry"]),
                "has_charges": len(mol_data["charges"]) > 0,
            }
        )

    return pd.DataFrame(summary)


def plot_homo_lumo_gap(df: pd.DataFrame, output_dir: str):
    """
    Create HOMO-LUMO gap plot.

    Args:
        df: DataFrame with molecule data
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 8))

    # Convert energies from atomic units to eV (1 a.u. = 27.211 eV)
    conversion = 27.211

    plt.scatter(df.index, df["homo"] * conversion, label="HOMO", color="blue", marker="o")
    plt.scatter(df.index, df["lumo"] * conversion, label="LUMO", color="red", marker="o")

    for i in range(len(df)):
        plt.plot(
            [i, i],
            [df["homo"].iloc[i] * conversion, df["lumo"].iloc[i] * conversion],
            color="gray",
            linestyle="--",
            alpha=0.6,
        )

    plt.xlabel("Molecule")
    plt.ylabel("Energy (eV)")
    plt.title("HOMO-LUMO Energies")
    plt.xticks(df.index, df["molecule_id"], rotation=90)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "homo_lumo_plot.png")
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved HOMO-LUMO plot to {output_file}")
    plt.close()


def plot_total_energies(df: pd.DataFrame, output_dir: str):
    """
    Create total energies plot.

    Args:
        df: DataFrame with molecule data
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 8))

    # Plot total energies in Hartree
    plt.bar(df.index, df["total_energy"], color="green", alpha=0.7)

    plt.xlabel("Molecule")
    plt.ylabel("Total Energy (Hartree)")
    plt.title("Total Energies")
    plt.xticks(df.index, df["molecule_id"], rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "total_energies_plot.png")
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved total energies plot to {output_file}")
    plt.close()


def create_molecule_images(data: List[Dict[str, Any]], output_dir: str):
    """
    Create molecule structure images with atom labels.

    Args:
        data: List of molecule data dictionaries
        output_dir: Directory to save images
    """
    os.makedirs(output_dir, exist_ok=True)

    for mol_data in data:
        # Create RDKit molecule from SMILES
        molecule_id = mol_data["molecule_id"]
        smiles = mol_data["smiles"]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Could not create molecule from SMILES for {molecule_id}")
                continue

            # Generate 2D coordinates for visualization
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)

            # Draw molecule
            img = Draw.MolToImage(mol, size=(400, 400), kekulize=True)

            # Save image
            output_file = os.path.join(output_dir, f"{molecule_id}_structure.png")
            img.save(output_file)
            logger.info(f"Saved molecule structure to {output_file}")

        except Exception as e:
            logger.error(f"Error creating molecule image for {molecule_id}: {str(e)}")


def generate_report(df: pd.DataFrame, output_dir: str):
    """
    Generate an HTML report with results summary.

    Args:
        df: DataFrame with molecule data
        output_dir: Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ORCA Quantum Chemistry Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            .figure img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>ORCA Quantum Chemistry Results</h1>
        
        <h2>Summary</h2>
        <p>Number of molecules: {len(df)}</p>
        
        <h2>Data Overview</h2>
        {df.to_html(index=False)}
        
        <h2>Visualizations</h2>
        <div class="figure">
            <h3>HOMO-LUMO Energies</h3>
            <img src="homo_lumo_plot.png" alt="HOMO-LUMO Energies">
        </div>
        
        <div class="figure">
            <h3>Total Energies</h3>
            <img src="total_energies_plot.png" alt="Total Energies">
        </div>
        
        <h2>Molecule Structures</h2>
        <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
    """

    # Add molecule images to the report
    for molecule_id in df["molecule_id"]:
        html_content += f"""
            <div style="margin: 10px; text-align: center; width: 30%;">
                <img src="{molecule_id}_structure.png" alt="{molecule_id}" style="max-width: 100%;">
                <p>{molecule_id}</p>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Write HTML report
    report_file = os.path.join(output_dir, "orca_results_report.html")
    with open(report_file, "w") as f:
        f.write(html_content)

    logger.info(f"Generated HTML report: {report_file}")


def main():
    """Main function to visualize ORCA results."""
    # Parse command line arguments
    args = parse_arguments()

    # Load ML data
    data = load_ml_data(args.results_dir)
    if not data:
        logger.error("No data to visualize")
        sys.exit(1)

    # Create summary DataFrame
    df = create_summary_dataframe(data)
    logger.info(f"Created summary for {len(df)} molecules")

    # Create visualizations based on plot_type
    if args.plot_type in ["all", "homo_lumo"]:
        plot_homo_lumo_gap(df, args.output_dir)

    if args.plot_type in ["all", "energies"]:
        plot_total_energies(df, args.output_dir)

    if args.plot_type in ["all", "structures"]:
        create_molecule_images(data, args.output_dir)

    # Generate report
    generate_report(df, args.output_dir)

    logger.info("Visualization completed successfully")


if __name__ == "__main__":
    main()
