import argparse
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import openmm
from openmm import unit as openmm_unit
from openmm import XmlSerializer
from openff.toolkit.topology import Molecule as OpenFFMolecule
from openff.toolkit.typing.engines.smirnoff import ForceField

from moml.models.mgnn.djmgnn import DJMGNN
from moml.data.feature_transforms import FeaturizeNodes
from torch_geometric.data import Data

def main():
    """
    Generates a molecule-specific OpenMM force field XML from a trained GNN checkpoint.
    This script follows the recipe provided to ensure correctness.
    """
    parser = argparse.ArgumentParser(description="Generate a force field from a trained model and SMILES string.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the molecule.")
    parser.add_argument("--out", type=str, required=True, help="Path to the output XML file.")
    args = parser.parse_args()

    # 1.A & 1.B: Load model and molecule
    print("Step 1: Loading model and molecule...")
    model = DJMGNN(in_node_dim=29, hidden_dim=128, n_blocks=3, layers_per_block=6, in_edge_dim=0, jk_mode="attention", node_output_dims=3, graph_output_dims=19, energy_output_dims=1, dropout=0.2, pool_type="mean", p_dropedge=0.1, use_supernode=True, use_rbf=True, rbf_K=32, env_dim=0, env_mlp=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # 1.C: Build PyG Data object
    print("Step 2: Featurizing molecule for GNN...")
    atoms = mol.GetAtoms()
    z = torch.tensor([atom.GetAtomicNum() for atom in atoms], dtype=torch.long)
    conf = mol.GetConformer()
    pos = torch.tensor([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(len(atoms))], dtype=torch.float)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)

    data = Data(z=z, pos=pos, edge_index=edge_index)
    data = FeaturizeNodes()(data)

    # 1.D: Run the GNN
    print("Step 3: Running GNN inference...")
    with torch.no_grad():
        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        dist = torch.norm(data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]], p=2, dim=-1).view(-1, 1)
        node_preds = model(x=data.x, edge_index=data.edge_index, batch=batch, dist=dist)['node_pred']

    raw_charges = node_preds[:, 0].numpy()
    # Crucially, neutralize the charges so they sum to the molecule's formal charge (0)
    net_charge = np.sum(raw_charges)
    formal_charge = Chem.GetFormalCharge(mol)
    charge_correction = (net_charge - formal_charge) / mol.GetNumAtoms()
    charges = raw_charges - charge_correction
    print(f"  Net charge before correction: {net_charge:.4f}, after: {np.sum(charges):.4f}")

    # Units from GNN are kcal/mol and Angstrom
    epsilons_kcal = node_preds[:, 1].numpy()
    sigmas_angstrom = node_preds[:, 2].numpy()

    # 1.E: Create a template OpenMM system
    print("Step 4: Creating template OpenMM system...")
    offmol = OpenFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)
    
    # **CRITICAL STEP**: Assign the GNN's neutralized charges to the molecule
    # BEFORE creating the system. This prevents the toolkit from trying to
    # calculate them with an external tool.
    offmol.partial_charges = openmm_unit.Quantity(charges, openmm_unit.elementary_charge)

    # Use an unconstrained force field to get the right functional forms
    ff = ForceField("openff_unconstrained-2.1.0.offxml")
    system = ff.create_openmm_system(offmol.to_topology())

    # 1.F: Overwrite nonbonded parameters in the OpenMM system
    print("Step 5: Overwriting nonbonded parameters with GNN predictions...")
    nonbonded_force = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]

    # Convert GNN outputs to OpenMM units (kJ/mol and nm)
    KJ_PER_KCAL = 4.184
    NM_PER_ANGSTROM = 0.1

    for i in range(mol.GetNumAtoms()):
        charge_i = charges[i]
        # OpenMM sigma is specified as distance, so it must be positive.
        # The GNN may predict negative values, so we take the absolute value.
        sigma_i = np.abs(sigmas_angstrom[i]) * NM_PER_ANGSTROM
        # Epsilon must also be positive.
        epsilon_i = np.abs(epsilons_kcal[i]) * KJ_PER_KCAL
        
        nonbonded_force.setParticleParameters(i, charge_i, sigma_i, epsilon_i)

    # 1.F: Write XML
    print(f"Step 6: Serializing final system to {args.out}...")
    with open(args.out, "w") as f:
        f.write(XmlSerializer.serialize(system))

    # 1.G: Quick self-test
    print("Step 7: Performing quick vacuum simulation smoke test...")
    try:
        integrator = openmm.LangevinIntegrator(300 * openmm_unit.kelvin, 1 / openmm_unit.picosecond, 2 * openmm_unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName('Reference')
        simulation = openmm.app.Simulation(offmol.to_topology().to_openmm(), system, integrator, platform)
        simulation.context.setPositions(offmol.conformers[0].to_openmm())
        simulation.minimizeEnergy()
        simulation.step(100)
        print("  Smoke test PASSED. XML is likely valid.")
    except Exception as e:
        print(f"  Smoke test FAILED: {e}")
        print("  The generated XML file may be invalid.")

if __name__ == "__main__":
    main()