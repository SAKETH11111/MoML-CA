import argparse
import openmm
from openmm import unit
from openmm.app import PDBFile, StateDataReporter, DCDReporter, Modeller, ForceField
from openff.toolkit.topology import Molecule

def main():
    """
    Sets up and runs a solvated OpenMM simulation for a given molecule
    using a custom GNN-generated force field XML file.
    """
    parser = argparse.ArgumentParser(description="Run a solvated OpenMM simulation.")
    parser.add_argument("--ff", type=str, required=True, help="Path to the force field XML file.")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the solute molecule.")
    parser.add_argument("--nsteps", type=int, required=True, help="Number of simulation steps.")
    parser.add_argument("--output_prefix", type=str, default="solvated", help="Prefix for output files.")
    parser.add_argument("--platform", type=str, default="CUDA", help="OpenMM platform to use (CUDA, OpenCL, CPU, Reference).")
    args = parser.parse_args()

    # 1. Load the molecule and generate a conformer
    print("Step 1: Loading molecule...")
    molecule = Molecule.from_smiles(args.smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)
    
    # Save a PDB file for visualization and topology loading
    pdb_filename = "solute.pdb"
    molecule.to_file(pdb_filename, file_format="PDB")
    pdb = PDBFile(pdb_filename)

    # 2. Load the serialized OpenMM System object from the XML file.
    print(f"Step 2: Loading serialized System from {args.ff}...")
    with open(args.ff, 'r') as f:
        system = openmm.XmlSerializer.deserialize(f.read())

    # 3. Set up modeller to add solvent
    print("Step 3: Adding solvent...")
    modeller = Modeller(pdb.topology, pdb.positions)
    water_ff = ForceField('amber14-all.xml')  # Use amber14 for TIP3P water
    modeller.addSolvent(water_ff, model='tip3p', padding=2.0*unit.nanometer)

    # Since the system is loaded from XML, we can't directly use it with a new topology
    # We need to create a new system combining the custom parameters with solvent
    # This is a limitation; for now, assume system is for solute only and needs adjustment
    # As a workaround, we'll use the topology with solvent but this may not work with custom FF
    print("Warning: Custom force field from XML may not apply correctly to solvated system. This is a placeholder.")
    combined_system = water_ff.createSystem(
        modeller.topology,
        nonbondedMethod=openmm.app.PME,
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=openmm.app.HBonds,
        rigidWater=True
    )
    # Note: The custom system parameters are not used here; this is a limitation to be fixed

    # 4. Set up the simulation
    print("Step 4: Setting up simulation...")
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    
    try:
        platform = openmm.Platform.getPlatformByName(args.platform)
    except openmm.OpenMMException:
        print(f"Warning: Platform '{args.platform}' not found. Using 'Reference' platform.")
        platform = openmm.Platform.getPlatformByName('Reference')

    simulation = openmm.app.Simulation(modeller.topology, combined_system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # 5. Minimize energy and run the simulation
    print("Step 5: Minimizing energy...")
    simulation.minimizeEnergy()

    # 6. Equilibration under NPT
    print("Step 6: Equilibrating under NPT...")
    combined_system.addForce(openmm.app.MonteCarloBarostat(1*unit.atmospheres, 300*unit.kelvin, 25))
    simulation.context.reinitialize(preserveState=True)
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    simulation.integrator = integrator
    simulation.step(125000)  # 250 ps at 2 fs timestep

    # 7. Production run under NVT
    print("Step 7: Running NVT production simulation...")
    # Remove barostat for NVT
    for force in combined_system.getForces():
        if isinstance(force, openmm.app.MonteCarloBarostat):
            combined_system.removeForce(combined_system.getForces().index(force))

    simulation.context.reinitialize(preserveState=True)
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    simulation.integrator = integrator

    # Set up reporters
    output_interval = 5000
    simulation.reporters.append(
        StateDataReporter(
            f"{args.output_prefix}_sim_data.csv",
            output_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=args.nsteps
        )
    )
    simulation.reporters.append(DCDReporter(f"{args.output_prefix}_trajectory.dcd", output_interval))
    
    print(f"\nStarting simulation for {args.nsteps} steps...")
    simulation.step(args.nsteps)

    print("\nSimulation finished.")
    state = simulation.context.getState(getEnergy=True)
    print(f"Final potential energy: {state.getPotentialEnergy()}")

if __name__ == "__main__":
    main() 