import torch
import numpy as np
import os

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
   def __init__(self, model, unit):
      super().__init__()
      self.model: torch.nn.Module = model
      self.unit: float = unit

   def forward(self, positions: torch.Tensor):

      positions = positions * self.unit
      
      potential = self.model(positions, True)

      return potential / self.unit
   


