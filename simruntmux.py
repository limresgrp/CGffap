import torch
from openmmtorch import TorchForce
import os

import yaml
from pathlib import Path


import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
# from force_mapper import ForceMapper

# import mdtraj as md

import training_modules as tm
from sys import stdout

from simutils import ForceReporter, ForceModelConvert

current_dir = os.getcwd()
fmartip_dir = os.path.dirname(current_dir)
dataset_dir = os.path.dirname(fmartip_dir)
output_file = os.path.join(dataset_dir, "enere@usi.ch/CGffap/dataset.VoidNoPBC.npz")  #fmartip/ff-naive/DatasetsA2A/dataset.A2a.hydrogen.pose0.npz
dataset = dict(np.load(output_file))
# Prepare a simulation system
atomic_numbers = dataset['atom_types']

index = 0
# bead mass calculation
bead_mass_dict = []
name_index = 0

for atoms in dataset["bead2atom_idcs"]:
    # print(i)
    bead_mass = 0
    num_atoms = 0
    for j in atoms:
        atom_name = dataset['atom_names'][j]
        if  "C" in atom_name:
            atom_mass = 12
            bead_mass += 12
        elif "O" in atom_name:
            atom_mass = 16
            bead_mass += 16
        elif "N" in atom_name:
            atom_mass = 14
            bead_mass += 14
        elif "S" in atom_name:
            atom_mass = 32
            bead_mass += 32
        elif "H" in atom_name:
            atom_mass = 1
            bead_mass += 1
        num_atoms += 1
    bead_mass_dict.append(bead_mass)
    # force_matrix[index] = force_matrix[index] / bead_mass * num_atoms
    name_index += 1
    index += 1
dataset['bead_mass'] = bead_mass_dict


pdb_file = os.path.join(dataset_dir,  "/home/enere@usi.ch/CGffap/ChigStartingCG.pdb" ) #Chignolin_CG_Unfolded  A2A-CG.pdb ChigStartingCG.pdb#'/home/enere@usi.ch/FMartIP/original_CG_A2A.pdb' "ChignCG_unfolded.pdb" "original_CG_A2A.pdb" "chig_CG/original_CG_a2a_Water.pdb" 
# "/home/enere@usi.ch/FMartIP/chig_CG/original_CG_a2a_4.pdb"
pdb = PDBFile(pdb_file) # OpenMM loader

bead_mass_dict = {}
name_index = 0
for mass in dataset['bead_mass']:
    bead_mass_dict[dataset['bead_idnames'][name_index]] = mass
    name_index += 1


index = 0
for atom, bead in zip(pdb.topology.atoms(), np.unique(dataset['bead_idnames'])):
    # print(chr(index + 150))
    i = dataset['bead_types'][np.where(dataset['bead_idnames'] == bead)]
    print(i[0]+100)
    mass = bead_mass_dict[bead]
    print(mass*amu)
    print(bead)
    try:
        atom.element = Element(number = i[0]+50, name = bead, symbol = str(index), mass = mass*amu)
    except:
        atom.element = Element.getByAtomicNumber(i[0]+50)
    index +=1
    print(Element.getByAtomicNumber(i[0]+50))
    print(chr(index))


bead_charges = []
for bead in dataset['bead_names']:
    if bead == 'GLU_SC1' or bead == 'ASP_SC1' :
        bead_charges.append(-1)
    elif bead == 'ARG_SC2' or bead == 'HIS_SC3' or bead == 'LYS_SC2':
        bead_charges.append(+1)
    else:
        bead_charges.append(0)
dataset['bead_charges'] = np.asanyarray(bead_charges).reshape(-1,1)


system = System()

for atom in pdb.topology.atoms():
    # print(atom)
    # print(dataset['bead_mass'][atom.index])
    system.addParticle(atom.element.mass)

# boxVectors = pdb.topology.getPeriodicBoxVectors()
# if boxVectors is not None:
#     system.setDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2])
# print(boxVectors)
system.usesPeriodicBoundaryConditions()


model_name = 'ChigAllFramesDatasetTest50.pt'
force = TorchForce('/home/enere@usi.ch/CGffap/Models/' + model_name) #'/home/enere@usi.ch/CGffap/bestmodelchigtestboard.pt'
# #

# I would need to still create the empty system
# forcefield = ForceField('amber14-all.xml')
# system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=None)
# integrator = LangevinIntegrator(300*kelvin, 1/picoseconds , 0.005 * picoseconds)

integrator = NoseHooverIntegrator(300*kelvin, 1/picosecond, 0.010*picoseconds) #VerletIntegrator( 0.010*picoseconds) #
#
#NoseHooverIntegrator(300*kelvin, 1/picosecond, 0.010*picoseconds)


while system.getNumForces() > 0:
    system.removeForce(0)
    
# The system should not contain any additional force and constrains
assert system.getNumConstraints() == 0
assert system.getNumForces() == 0

# Add the NNP to the system
system.addForce(force)

# This line combines the molecular topology, system, and integrator to begin a new simulation. It creates a Simulation object and assigns it to a variable called simulation. 
# A Simulation object manages all the processes involved in running a simulation, such as advancing time and writing output.
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.getPositions())

# Performs a local energy minimization. It is usually a good idea to do this at the start of a simulation, since the coordinates in the PDB file might produce very large forces.
# simulation.minimizeEnergy()
print("starting Sim")

simulation.reporters.append(PDBReporter( 'Sims/' + model_name + 'tmux.pdb', 100))
simulation.reporters.append(StateDataReporter('Sims/' + model_name + 'tmux.dat', 100, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, time=True, totalEnergy=True))
simulation.reporters.append(ForceReporter('outputforces.txt', 100))

#This line adds another reporter to print out some basic information every 1000 time steps
simulation.step(1000000)
state = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
f = np.array([[a.x,a.y,a.z]for a in state.getForces()])
p = np.array([[a.x,a.y,a.z]for a in state.getPositions()])
# print(state.getForces(), state.getPositions())