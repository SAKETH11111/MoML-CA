"""
Compress polymer slabs to target density using NPT simulation.
"""

from openmm import app, unit, LangevinIntegrator, Platform, MonteCarloBarostat, State
import numpy as np

def compress(topology, positions, forcefield, target_density, steps=50000):
    """
    Quick 100-ps 10-bar NPT to pre-compress flexible membranes / resins.
    Returns new positions.
    
    Args:
        topology: OpenMM topology
        positions: Initial positions
        forcefield: OpenMM forcefield
        target_density: Target density in g/cm³
        steps: Number of simulation steps
        
    Returns:
        numpy array of new positions
    """
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME,
                                   constraints=app.HBonds, rigidWater=True)
    system.addForce(MonteCarloBarostat(10*unit.bar, 300*unit.kelvin, 25))
    integrator = LangevinIntegrator(600*unit.kelvin, 1/unit.picosecond,
                                  0.002*unit.picoseconds)
    platform = Platform.getPlatformByName("CPU")  # fast startup
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    sim.minimizeEnergy()

    sim.reporters.append(app.StateDataReporter(None, 5000, density=True))
    sim.step(steps)

    # cool to 300 K
    integrator.setTemperature(300*unit.kelvin)
    sim.step(int(steps/3))

    # Get final positions
    state = sim.context.getState(getPositions=True)
    return state.getPositions(asNumpy=True) 