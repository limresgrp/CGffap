import time
import numpy as np
import MDAnalysis as mda
from openmm.app import StateDataReporter
from MDAnalysis.core._get_readers import get_reader_for

class TrajectoryReporter(StateDataReporter):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hasInitialized = False
        self._needsPositions = True
        self._needsVelocities = False
        self._needsForces = True

        self._values = []
        self._positions = []
        self._forces = []

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._hasInitialized:
            self._initializeConstants(simulation)
            headers = self._constructHeaders()
            if not self._append:
                print('#"%s"' % ('"'+self._separator+'"').join(headers), file=self._out)
            try:
                self._out.flush()
            except AttributeError:
                pass
            self._initialClockTime = time.time()
            self._initialSimulationTime = state.getTime()
            self._initialSteps = simulation.currentStep
            self._hasInitialized = True

        # Check for errors.
        self._checkForErrors(simulation, state)

        # Query for the values
        values = self._constructReportValues(simulation, state)

        # Write the values.
        print(self._separator.join(str(v) for v in values), file=self._out)

        p = np.array([[a.x,a.y,a.z] for a in state.getPositions()])
        f = np.array([[a.x,a.y,a.z] for a in state.getForces()])

        self._positions.append(p)
        self._forces.append(f)

        try:
            self._out.flush()
        except AttributeError:
            pass
    
    def dump(self, input_filename, output_filename):
        u = mda.Universe(input_filename)
        coords = np.stack(self._positions, axis=0)
        forces = np.stack(self._forces, axis=0)
        u.trajectory = get_reader_for(coords)(
                coords, order='fac', n_atoms=u.atoms.n_atoms,
                velocities=None, forces=forces)
        sel = u.select_atoms('all')
        with mda.Writer(output_filename, n_atoms=sel.n_atoms) as w:
            for ts in u.trajectory:
                w.write(sel)