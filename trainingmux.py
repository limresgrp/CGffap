import torch
import torch.nn as nn
from openmmtorch import TorchForce
import os

import yaml
from pathlib import Path


import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *


from sys import stdout

from maputils import EquiValReporter

import training_modules as tm

from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import matplotlib.pyplot as plt
from torch.utils.data import Dataset , DataLoader
from training_utils import CGDataset , TrainSystem
import os
import yaml
import numpy as np
from pathlib import Path
from cgmap.utils import DataDict
from cgmap.mapping import Mapper

from typing import Dict

from aggforce import linearmap as lm
from aggforce import agg as ag
from aggforce import constfinder as cf


dataset = dict(np.load('dataset.NoPBCVoidALLFrames.npz', allow_pickle=True))

equvalrep = EquiValReporter(dataset=dataset)

equvalrep.bondMapper(config_file_path="config/bond_config.yaml")
equvalrep.angleMapper(conf_angles_path="test_conf/config.angles.yaml")
equvalrep.improperDihedralMapper(conf_angles_path="config/config.naive.per.bead.type.dihedrals.yaml")
equvalrep.beadChargeMapper()

dataset = equvalrep.getDataset()

conf_bonds: dict = equvalrep.getBonds()
conf_angles: dict = equvalrep.getAngles()
conf_dihedrals: dict = equvalrep.getImproperDihs()
conf_bead_charges: dict = equvalrep.getBeadCharges()

system = TrainSystem(dataset, conf_bonds, conf_angles, conf_dihedrals, conf_bead_charges)#, device_index=1

model = system.initiateTraining(dataset = dataset, train_steps=800, batch_size=128, patience=20, model_name='ChigAllFramesHbondSelectedTmux')

print('stopped')
