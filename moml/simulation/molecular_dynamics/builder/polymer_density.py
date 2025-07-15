"""
Compress polymer slabs to target density using NPT simulation.
"""

import sys 
from openmm import app, unit, LangevinIntegrator, Platform, MonteCarloBarostat, State
import numpy as np

import logging

logger = logging.getLogger(__name__)

def compress(topology, positions, forcefield, target_density, steps=50000, report_interval=1000):
    """
    Compress polymer slabs to target density using NPT simulation with early termination.
    Returns new positions.
    
    Args:
        topology: OpenMM topology
        positions: Initial positions
        forcefield: OpenMM forcefield
        target_density: Target density in g/cm³
        steps: Maximum number of simulation steps
        report_interval: Interval for reporting density and checking termination
        
    Returns:
        numpy array of new positions
    """
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME,
                                   constraints=app.HBonds, rigidWater=True)
    system.addForce(MonteCarloBarostat(10 * unit.bar, 300 * unit.kelvin, 25))
    integrator = LangevinIntegrator(600 * unit.kelvin, 1 / unit.picoseconds,
                                  0.002 * unit.picoseconds)
    platform = Platform.getPlatformByName("CPU")  # fast startup
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    sim.minimizeEnergy()

    # Add a reporter to get density
    sim.reporters.append(app.StateDataReporter(sys.stdout, report_interval, density=True, totalSteps=steps, speed=True, remainingTime=True, separator='\t'))

    current_density = 0.0 * unit.gram / (unit.centimeter**3)
    for i in range(0, steps, report_interval):
        sim.step(report_interval)
        state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        volume = state.getPeriodicBoxVectors()[0][0] * state.getPeriodicBoxVectors()[1][1] * state.getPeriodicBoxVectors()[2][2]
        mass = system.getMass(0) * topology.getNumAtoms() # Assuming all atoms have same mass for simplicity, or sum actual masses
        # A more accurate mass calculation would iterate through all atoms:
        total_mass = sum(system.getParticleMass(j) for j in range(system.getNumParticles()))
        current_density = total_mass / volume

        logger.info(f"Step {i+report_interval}/{steps}: Current Density = {current_density:.3f}, Target Density = {target_density:.3f}")

        if current_density >= target_density:
            logger.info(f"Target density {target_density:.3f} reached at step {i+report_interval}. Terminating compression.")
            break

    # cool to 300 K
    integrator.setTemperature(300 * unit.kelvin)
    sim.step(int(steps/3))

    # Get final positions
    state = sim.context.getState(getPositions=True)
    return state.getPositions(asNumpy=True)