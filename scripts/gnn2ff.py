"""
scripts/gnn2ff.py

This script generates a molecule-specific OpenMM force field XML file from a
trained Graph Neural Network (GNN) model. It takes a model checkpoint and a
SMILES string as input, uses the GNN to predict atomic parameters (partial
charges, Lennard-Jones sigma and epsilon), and then creates an OpenMM `System`
object with these parameters.

The process is as follows:
1.  Load a trained DJMGNN model from a checkpoint.
2.  Generate a 3D conformer for the input SMILES string.
3.  Featurize the molecule into a PyTorch Geometric `Data` object.
4.  Run GNN inference to get atom-level predictions.
5.  Neutralize the predicted partial charges to match the molecule's formal charge.
6.  Create a template OpenMM system using an unconstrained OpenFF force field.
7.  Overwrite the nonbonded parameters (charge, sigma, epsilon) in the system
    with the GNN's predictions.
8.  Serialize the final, parameterized system to an XML file.
9.  Perform a quick "smoke test" simulation to check for immediate errors.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import openmm
import torch
from openff.toolkit.topology import Molecule as OpenFFMolecule  # type: ignore
from openff.toolkit.typing.engines.smirnoff import ForceField  # type: ignore
from openmm import XmlSerializer, unit as openmm_unit
from openmm import app
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.feature_transforms import FeaturizeNodes
from moml.models.mgnn.djmgnn import DJMGNN

DEFAULT_NODE_FEATURE_DIM = 29
KJ_PER_KCAL = 4.184
NM_PER_ANGSTROM = 0.1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(ckpt_path: str) -> DJMGNN:
    """Load a DJMGNN model from a checkpoint file."""
    logger.info(f"Loading model from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    
    # Use a default config if not found in checkpoint, which might be the case for older models.
    model_config = checkpoint.get("model_config", {})
    if not model_config:
        logger.warning("No 'model_config' in checkpoint. Using default hyperparameters.")

    model = DJMGNN(
        in_node_dim=model_config.get("in_node_dim", DEFAULT_NODE_FEATURE_DIM),
        hidden_dim=model_config.get("hidden_dim", 128),
        n_blocks=model_config.get("n_blocks", 4),
        in_edge_dim=model_config.get("in_edge_dim", 0),
        node_output_dims=model_config.get("node_output_dims", 3),
        graph_output_dims=model_config.get("graph_output_dims", 19),
        energy_output_dims=model_config.get("energy_output_dims", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def create_molecule_from_smiles(smiles: str) -> Chem.Mol:
    """Create and embed an RDKit molecule from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not parse SMILES string: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1:  # type: ignore
        raise RuntimeError(f"Could not generate a 3D conformer for: {smiles}")
    return mol


def featurize_molecule(mol: Chem.Mol) -> Data:
    """Convert an RDKit molecule to a PyG Data object."""
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    data = Data(z=z, pos=pos, edge_index=edge_index, mol=mol)
    return FeaturizeNodes()(data)


def run_gnn_inference(model: DJMGNN, data: Data) -> np.ndarray:
    """Run GNN inference to get node-level predictions."""
    if data.x is None or data.pos is None or data.edge_index is None:
        raise ValueError("Input data is missing node features, positions, or edge indices.")

    with torch.no_grad():
        batch_map = torch.zeros(data.x.size(0), dtype=torch.long)
        dist = torch.norm(data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]], p=2, dim=-1)
        out = model(x=data.x, edge_index=data.edge_index, batch=batch_map, dist=dist)
    
    if "node_pred" not in out or out["node_pred"] is None:
        raise ValueError("GNN model did not return 'node_pred'.")

    node_preds = out["node_pred"].numpy()
    if len(node_preds) != data.mol.GetNumAtoms():
        raise ValueError("GNN output size does not match the number of atoms.")
    return node_preds


def create_openmm_system(
    mol: Chem.Mol, gnn_preds: np.ndarray, ff_xml: str = "openff_unconstrained-2.1.0.offxml"
) -> openmm.System:
    """Create a parameterized OpenMM system from GNN predictions."""
    formal_charge = Chem.GetFormalCharge(mol)
    raw_charges = gnn_preds[:, 0]
    charge_correction = (np.sum(raw_charges) - formal_charge) / mol.GetNumAtoms()
    charges = raw_charges - charge_correction
    logger.info(f"Net charge corrected from {np.sum(raw_charges):.4f} to {np.sum(charges):.4f}")

    epsilons_kcal = np.abs(gnn_preds[:, 1])
    sigmas_angstrom = np.abs(gnn_preds[:, 2])
    
    offmol = OpenFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)
    offmol.partial_charges = openmm_unit.Quantity(charges, openmm_unit.elementary_charge)

    forcefield = ForceField(ff_xml)
    system = forcefield.create_openmm_system(offmol.to_topology())
    nonbonded_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))

    for i in range(mol.GetNumAtoms()):
        nonbonded_force.setParticleParameters(
            i,
            charges[i],
            sigmas_angstrom[i] * NM_PER_ANGSTROM,
            epsilons_kcal[i] * KJ_PER_KCAL,
        )
    return system


def run_smoke_test(system: openmm.System, offmol: OpenFFMolecule):
    """Run a brief simulation to test the validity of the generated system."""
    try:
        integrator = openmm.LangevinIntegrator(300 * openmm_unit.kelvin, 1 / openmm_unit.picosecond, 0.002 * openmm_unit.picoseconds)  # type: ignore
        platform = openmm.Platform.getPlatformByName("Reference")
        simulation = app.Simulation(offmol.to_topology().to_openmm(), system, integrator, platform)
        simulation.context.setPositions(offmol.conformers[0].to_openmm())
        simulation.minimizeEnergy()
        simulation.step(100)
        logger.info("Smoke test PASSED. The generated XML appears to be valid.")
    except Exception as e:
        logger.error(f"Smoke test FAILED: {e}", exc_info=True)


def main():
    """Main function to generate a force field from a GNN model."""
    parser = argparse.ArgumentParser(
        description="Generate a force field from a trained model and SMILES string."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the molecule.")
    parser.add_argument("--out", type=str, required=True, help="Path to the output XML file.")
    args = parser.parse_args()

    try:
        # 1. Load model and molecule
        model = load_model_from_checkpoint(args.ckpt)
        rdkit_mol = create_molecule_from_smiles(args.smiles)

        # 2. Featurize and run inference
        data = featurize_molecule(rdkit_mol)
        gnn_preds = run_gnn_inference(model, data)

        # 3. Create and parameterize the OpenMM system
        system = create_openmm_system(rdkit_mol, gnn_preds)

        # 4. Write XML output
        logger.info(f"Serializing final system to {args.out}...")
        with open(args.out, "w") as f:
            f.write(XmlSerializer.serialize(system))

        # 5. Run a self-test
        offmol = OpenFFMolecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
        run_smoke_test(system, offmol)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()