import torch
import numpy as np

from typing import Dict
from pathlib import Path
from cgmap.mapping.mapper import Mapper

from openmmtorch import TorchForce
from openmm.app import PDBFile, Element
from openmm import System

class ForceReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.forces = []

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):

        # system = simulation.context.getState(getForces=True)
        f = [[a.x,a.y,a.z] for a in state.getForces()]
        # mass = system.getParticleMass(1).value_in_unit(dalton)
        self._out.write(str(f))
        self.forces.append(f)

    def getForces(self):

        return np.array(self.forces)


class ForceModelConvert(torch.nn.Module):
    def __init__(
        self,
        model,
        bead_types: torch.Tensor,
        pos2unit: float,
        energy2unit: float,
    ):
        super().__init__()
        self.model: torch.nn.Module = model
        self.model.set_bead_types(bead_types)
        self.pos2unit: float = pos2unit
        self.energy2unit: float = energy2unit

    def forward(self, positions: torch.Tensor):
        positions = positions * self.pos2unit
        potential = self.model(positions)
                
        # - Use to check consistency with training - #
        
        # grads = torch.autograd.grad(
        # [potential],
        # [positions],
        # create_graph=True,  # needed to allow gradients of this output during training
        # )

        # print(grads[0])

        # ------------------------------------------ #

        return potential * self.energy2unit


def build_system(args_dict: Dict):

    # - Either map atomistic input file to CG or parse CG file - #
    
    mapping = Mapper(args_dict=args_dict)
    mapping.map(index=0)
    dataset = mapping.dataset

    if args_dict.get('isatomistic', False):
        inputcg = args_dict.get('inputcg', None)
        if inputcg is None:
            p = Path(args_dict.get('input'))
            inputcg = str(Path(p.parent, p.stem + '.CG' + p.suffix))
            args_dict['inputcg'] = inputcg
        mapping.save(filename=inputcg)
    else:
        args_dict['inputcg'] = args_dict.get('input')

    # - Compute the mass of each bead - #

    bead_mass_dict = []
    for atoms in dataset["bead2atom_idcs"]:
        bead_mass = None
        for j in atoms[atoms > -1]:
            atom_type = dataset['atom_types'][j]
            if bead_mass is None:
                bead_mass = Element._elements_by_atomic_number[atom_type].mass
            else:
                bead_mass += Element._elements_by_atomic_number[atom_type].mass
        bead_mass_dict.append(bead_mass)
    dataset['bead_mass'] = bead_mass_dict
    bead_mass_dict = {}
    for idname, bead_mass in zip(dataset['bead_idnames'], dataset['bead_mass']):
        bead_mass_dict[idname] = bead_mass
    
    # - Load CG pdb file into OpenMM - #
    
    pdb = PDBFile(args_dict.get('inputcg'))

    # - Create one Element object for each bead type, assigning the bead mass to it - #

    starting_atomic_number = 200
    unique_bead_idnames = np.unique(dataset['bead_idnames'])
    for atom, bead_idname in zip(pdb.topology.atoms(), np.unique(dataset['bead_idnames'])):
        i = dataset['bead_types'][np.where(unique_bead_idnames == bead_idname)][0]
        mass = bead_mass_dict[bead_idname]
        atomic_number = starting_atomic_number + i
        symbol = str(i)
        try:
            atom.element = Element.getByAtomicNumber(atomic_number)
        except:
            atom.element = Element(
                number=atomic_number,
                name=bead_idname,
                symbol=symbol,
                mass=mass
            )

    # - Create System and add beads as particles - #

    system = System()

    for atom in pdb.topology.atoms():
        system.addParticle(atom.element.mass)
    
    # boxVectors = pdb.topology.getPeriodicBoxVectors()
    # if boxVectors is not None:
    #     system.setDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2])
    # print(boxVectors)
    # system.usesPeriodicBoundaryConditions()
    
    # - Ensure that the system does not contain any force or constraint - #
    
    while system.getNumForces() > 0:
        system.removeForce(0)
        
    assert system.getNumConstraints() == 0
    assert system.getNumForces() == 0

    # - Wrap trained module to interface with OpenMM            - #
    # - Convert model unit of measure to OpenMM unit of measure - #
    
    trained_module_filename = args_dict.get('model')
    trained_module = torch.jit.load(trained_module_filename)
    wrapper_module = ForceModelConvert(
        trained_module,
        bead_types=torch.from_numpy(dataset['bead_types']),
        pos2unit=args_dict.get('pos2unit', 1.0),
        energy2unit=args_dict.get('energy2unit', 1.0),
    )
    wrapper_module.to('cpu')
    ff_module = torch.jit.script(
        wrapper_module
    )

    p = Path(trained_module_filename)
    ff_module_filename = str(Path(p.parent, p.stem + '.ff' + p.suffix))
    ff_module.save(ff_module_filename)

    # - Load and assign the wrapped force field model to the system - #

    ff = TorchForce(ff_module_filename)
    system.addForce(ff)
    
    return system, pdb