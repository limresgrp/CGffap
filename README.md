# Coarse-Grained Force Field Automatic Parametrisation (CGffap)

CGffap is a parametrisation tool provides optimal parameters for CG beads given a fine-grained simulation. 

## Installation 

CGffap utilises a couple other tools in tandem to function, namely [CGmap](https://github.com/limresgrp/CGmap) and [OpenMM-torch](https://github.com/openmm/openmm-torch). 

CGmap allows us to create CG mappings for all-atom simulations, using any mapping we prefer. While OpenMM-torch enables us to replace the potential energy function of the simulaation software with the ones we have implemented. 

It is important to note that the cuda version compatability is crucial for everrything to work properly and the use of OpenMM with cuda enabled. 