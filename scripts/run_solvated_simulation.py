import os
import sys
import argparse
import numpy as np
from openmm import app, unit, LangevinIntegrator, Platform
from openmm.app import forcefield as ff
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import xml.etree.ElementTree as ET
from openmm import XmlSerializer

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run solvated MD simulation for PFAS molecules')
parser.add_argument('--molecule', type=str, required=True, help='Molecule name (e.g., pfos or pfoa)')
parser.add_argument('--xml_file', type=str, required=True, help='Path to the force field XML file')
parser.add_argument('--structure_file', type=str, required=True, help='Path to the structure file (PDB or SDF) for the molecule')
parser.add_argument('--steps', type=int, default=2500000, help='Number of simulation steps (default: 2.5M steps = 5 ns)')
parser.add_argument('--output_dir', type=str, default='analysis_outputs', help='Directory for output files')
args = parser.parse_args()

# Set up paths
molecule = args.molecule.lower()
xml_file = args.xml_file
structure_file = args.structure_file
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Load the force field
# Since the XML is a system definition, load it directly using XmlSerializer
with open(xml_file, 'r') as f:
    xml_content = f.read()
    if '<System' in xml_content:
        with open(xml_file, 'r') as system_file:
            system = XmlSerializer.deserialize(system_file.read())
        # Create a topology from the system (this is a placeholder, may need adjustment)
        topology = app.Topology()
        # Add chains, residues, and atoms as needed (this part needs to match the system)
        # For simplicity, assume the system has the topology embedded or can be used directly
        modeller = app.Modeller(topology, [])
        # We'll use the system directly for simulation
    else:
        raise ValueError(f'The file {xml_file} does not appear to be a valid OpenMM System XML.')

# Since we're loading the system directly, we need a standard force field for solvent
water_forcefield = ff.ForceField('amber14-all.xml')

# Add solvent (TIP3P water box with 2 nm padding)
modeller.addSolvent(water_forcefield, model='tip3p', padding=2.0*unit.nanometer)

# Set up integrator
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)

# Set up platform (use CUDA if available, else CPU)
platform = Platform.getPlatformByName('CUDA' if Platform.getPlatformByName('CUDA').getSpeed() > Platform.getPlatformByName('CPU').getSpeed() else 'CPU')

# Create simulation
# Use the loaded system instead of creating a new one
simulation = app.Simulation(modeller.topology, system, integrator, platform)
# Set positions (this is tricky since the system may have positions, but modeller may not)
# For now, assume positions are in the system or need to be set
# This part may need adjustment based on the XML content
simulation.context.setPositions(modeller.positions if modeller.positions else [unit.Vec3(0,0,0)] * system.getNumParticles())

# Minimize energy
print('Minimizing energy...')
simulation.minimizeEnergy(maxIterations=1000)

# Equilibration (NPT for 250 ps)
print('Equilibrating under NPT...')
system.addForce(app.MonteCarloBarostat(1*unit.atmospheres, 300*unit.kelvin, 25))
simulation.context.reinitialize(preserveState=True)
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
simulation.integrator = integrator
simulation.step(125000)  # 250 ps at 2 fs timestep

# Production run (NVT for 5 ns)
print('Running NVT production simulation...')
# Remove barostat for NVT
for force in system.getForces():
    if isinstance(force, app.MonteCarloBarostat):
        system.removeForce(system.getForces().index(force))

simulation.context.reinitialize(preserveState=True)
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
simulation.integrator = integrator

# Set up reporters
simulation.reporters.append(app.DCDReporter(f'{output_dir}/{molecule}_solvated_trajectory.dcd', 5000))
simulation.reporters.append(app.StateDataReporter(f'{output_dir}/{molecule}_solvated_sim_data.csv', 5000, step=True, potentialEnergy=True, temperature=True, volume=True))

# Run simulation
simulation.step(args.steps)

print(f'Simulation completed. Output saved to {output_dir}/{molecule}_solvated_sim_data.csv and {output_dir}/{molecule}_solvated_trajectory.dcd') 