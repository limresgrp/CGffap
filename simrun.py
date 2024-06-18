import os

from datetime import datetime
from pathlib import Path
from openmm.app import *
from openmm import *
from openmm.unit import kelvin, picoseconds

from simutils import ForceReporter, build_system

args_dict = {
    'mapping': 'martini3',
    'input': 'chignolin.data.test.CG.pdb',
    'selection': 'protein and not resname NME',
    'isatomistic': False,        # If the input pdb is an atomistic pdb, set this to True. It will save a CG version and load it in OpenMM
    'pos2unit': AngstromsPerNm,  # If the model works in Angstrom use this, if it works in nm just comment this line
    'energy2unit': KJPerKcal,    # If the model works in Kcal use this, if it works in KJ just comment this line
    'model': 'models/chignolin.A.kcal.best.pt', # Path to the compiled model saved after training
    'sim_folder': 'simulations', # Save simulation and logs inside this folder
    'log_every': 100,            # Log every N simulation steps
    'steps': 100000,              # Simulation steps
}

def main():
    system, pdb = build_system(args_dict)

    # integrator = LangevinIntegrator(310*kelvin, 1./picoseconds , 0.01*picoseconds)
    # integrator = VerletIntegrator(0.01*picoseconds)
    integrator = NoseHooverIntegrator(310*kelvin, 10./picoseconds, 0.010*picoseconds)
    platform = Platform.getPlatformByName('CPU')

    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.getPositions())

    # Create simulaiton folder - #

    p = Path(args_dict.get('model'))
    sim_root = os.path.join(args_dict.get('sim_folder', 'simulations'), str(p.stem), datetime.now().strftime("%Y.%m.%d-%H:%M"))
    os.makedirs(str(sim_root), exist_ok=True)

    # - Save initial structure to a PDB file inside simulation folder - #

    PDBFile.writeFile(pdb.topology, pdb.getPositions(), open(os.path.join(sim_root, 'initial.pdb'), 'w'))

    # Performs a local energy minimization.                                  #
    # It is usually a good idea to do this at the start of a simulation,     #
    # since the coordinates in the PDB file might produce very large forces. #

    print("Minimizing energy...")
    simulation.minimizeEnergy()

    # - Write minimized system to a PDB file - #
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(pdb.topology, minimized_positions, open(os.path.join(sim_root, 'minimized.pdb'), 'w'))

    # - Logging during simulation - #

    log_every = args_dict.get('log_every', 1000)
    simulation.reporters.append(PDBReporter(
        os.path.join(sim_root, 'output.pdb'), log_every
    ))
    simulation.reporters.append(StateDataReporter(
        os.path.join(sim_root, 'output.dat'), log_every, step=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, time=True, totalEnergy=True
    ))
    simulation.reporters.append(ForceReporter(
        os.path.join(sim_root, 'outputforces.txt'), log_every
    ))

    # - Start simulation - #

    print("Starting simulation...")
    simulation.step(args_dict.get('steps'))
    print("Simulation ended!")


if __name__ == "__main__":
    main()