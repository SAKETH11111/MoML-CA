import argparse
import openmm
from openmm import unit
from openmm.app import PDBFile, StateDataReporter, DCDReporter, Modeller, ForceField
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField as OpenFFForceField

def main():
    """
    Sets up and runs a vacuum or solvated OpenMM simulation for a given molecule
    using a custom GNN-generated force field XML file.
    """
    parser = argparse.ArgumentParser(description="Run an OpenMM simulation.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the solute molecule.")
    parser.add_argument("--forcefield", type=str, required=True, help="Comma-separated list of force field OFFXML files.")
    parser.add_argument("--nsteps", type=int, required=True, help="Number of simulation steps.")
    parser.add_argument("--platform", type=str, default="CUDA", help="OpenMM platform to use (CUDA, OpenCL, CPU, Reference).")
    parser.add_argument("--solvate", action="store_true", help="Whether to run a solvated simulation (default: False).")
    parser.add_argument("--padding", type=float, default=1.5, help="Padding in nm for solvent box (default: 1.5).")
    parser.add_argument("--ionic_strength", type=float, default=0.0, help="Salt concentration in mol/L for ionic strength (default: 0.0).")
    args = parser.parse_args()

    # 1. Load the molecule and generate a conformer
    print("Step 1: Loading molecule...")
    molecule = Molecule.from_smiles(args.smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    
    # Save a PDB file for visualization and topology loading
    pdb_filename = "solute.pdb"
    molecule.to_file(pdb_filename, file_format="PDB")
    pdb = PDBFile(pdb_filename)
    
    # 2. Load the force field
    print(f"Step 2: Loading force field from {args.forcefield}...")
    forcefield = OpenFFForceField(*args.forcefield.split(','))

    # 3. Set up modeller to add solvent if requested
    if args.solvate:
        print("Step 3: Adding solvent...")
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(forcefield, model='tip3p', padding=args.padding * unit.nanometer, ionicStrength=args.ionic_strength * unit.molar)
        topology, positions = modeller.getTopology(), modeller.getPositions()
        system = forcefield.create_openmm_system(topology, charge_from_molecules=[molecule])
        print("Solvent added successfully.")
    else:
        topology = pdb.topology
        positions = pdb.positions
        system = forcefield.create_openmm_system(topology)

    # 4. Set up the simulation
    print("Step 4: Setting up simulation...")
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    
    try:
        platform = openmm.Platform.getPlatformByName(args.platform)
    except openmm.OpenMMException:
        print(f"Warning: Platform '{args.platform}' not found. Using 'Reference' platform.")
        platform = openmm.Platform.getPlatformByName('Reference')

    simulation = openmm.app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # 5. Minimize energy and run the simulation
    print("Step 5: Minimizing energy...")
    simulation.minimizeEnergy()

    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    
    print(f"\nStarting simulation for {args.nsteps} steps...")
    output_interval = 5000
    simulation.reporters.append(
        StateDataReporter(
            "sim_data.csv",
            output_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=args.nsteps
        )
    )
    simulation.reporters.append(DCDReporter('trajectory.dcd', output_interval))
    
    simulation.step(args.nsteps)

    print("\nSimulation finished.")
    state = simulation.context.getState(getEnergy=True)
    print(f"Final potential energy: {state.getPotentialEnergy()}")

if __name__ == "__main__":
    main()
