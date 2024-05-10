from torch.utils.data import Dataset , DataLoader
import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch.nn as nn

import yaml
from pathlib import Path
from typing import List, Union, Optional, Dict

import numpy as np
import os

from openmmtorch import TorchForce
from openmm.app import *
from openmm import *
from openmm.unit import *


class ForceModel(torch.nn.Module):
   def __init__(self, dataset, conf_bonds: dict, conf_angles: dict, conf_dihedrals: dict, conf_bead_charges: dict, device_index = 0):
      super().__init__()
      torch.cuda.set_device(device_index)
      self.device = 'cuda'
      self.dataset = dataset

      # Define Indices
      bond_indices : np.ndarray = self.dataset['bond_indices']
      angle_indices = dataset['angle_indices']
      improper_dih_indices = dataset['improper_dih_indices']
      
      # Remove bonded indices form nonbonded
      # upper triangluar transform index finder given the indices of the bead first is smaller than the second bead index
      self.all_indices = np.array(np.triu_indices(len(dataset['bead_types']), 1))
      ones_array = torch.ones(len(dataset['bead_types']), len(dataset['bead_types'])).to(self.device)
      
      first_bead = bond_indices[:,0]
      second_bead = bond_indices[:,1]
      # anglebeads

      first_bead = np.append(first_bead,dataset["angle_indices"][:,0])
      second_bead = np.append(second_bead,dataset["angle_indices"][:,2])

      index = ((first_bead[:] * (first_bead[:] + 1)) / 2 + first_bead[:] * (len(dataset['bead_types']) - first_bead[:] - 1) + second_bead[:] - first_bead[:] - 1).astype(int)
      self.nonbonded_indices = [[],[]]
      self.nonbonded_indices[0] = np.delete(self.all_indices[0],index,None)
      self.nonbonded_indices[1] = np.delete(self.all_indices[1],index,None)
      

      self.nonbond_interactions = ones_array[self.nonbonded_indices[0],self.nonbonded_indices[1]]

      # NonBonded Interactions
      nonbonded_names = [[dataset['bead_idnames'][self.nonbonded_indices[0][i]],dataset['bead_idnames'][self.nonbonded_indices[1][i]]] for i in range(len(self.nonbonded_indices[0]))]
      nonbonded_names = np.array(nonbonded_names)
      self.nonbonded_dict = {}
      for i, j in nonbonded_names:
         if (str(i + "-" + j) and str(j + "-" + i)) not in self.nonbonded_dict.keys():
            self.nonbonded_dict[i + "-" + j] = 1

      self.nonbonded_keys = np.asanyarray([i for i in self.nonbonded_dict.keys()])
      self.nonbonded_vals = torch.from_numpy(np.asanyarray([i for i in self.nonbonded_dict.values()])).to(self.device)

      self.nonbonded_type_keys = np.unique(np.sort(np.array([[dataset['bead_types'][self.nonbonded_indices[0][i]],dataset['bead_types'][self.nonbonded_indices[1][i]]] for i in range(len(self.nonbonded_indices[0]))]), axis=1), axis=0)

      self.nonbond_dict_index = []
      self.nonbond_type_dict_index = []
      self.nonbonded_pairs = [[self.nonbonded_indices[0][i],self.nonbonded_indices[1][i]] for i in range(len(self.nonbonded_indices[0]))]

      for row in self.nonbonded_pairs:
         if str(self.dataset["bead_idnames"][row[0]] + '-' + self.dataset["bead_idnames"][row[1]]) in self.nonbonded_keys:
            key = str(self.dataset["bead_idnames"][row[0]] + '-' + self.dataset["bead_idnames"][row[1]]) 
         elif str(self.dataset["bead_idnames"][row[1]] + '-' + self.dataset["bead_idnames"][row[0]]) in self.nonbonded_keys:
            key = str(self.dataset["bead_idnames"][row[1]] + '-' + self.dataset["bead_idnames"][row[0]]) 

         if any([all([self.dataset["bead_types"][row[0]], self.dataset["bead_types"][row[1]]] == arr) for arr in self.nonbonded_type_keys]):
            index_key = np.array([self.dataset["bead_types"][row[0]], self.dataset["bead_types"][row[1]]])
         elif any([all([self.dataset["bead_types"][row[1]], self.dataset["bead_types"][row[0]]] == arr) for arr in self.nonbonded_type_keys]):
            index_key = np.array([self.dataset["bead_types"][row[1]], self.dataset["bead_types"][row[0]]])

         self.nonbond_dict_index.append(np.where(self.nonbonded_keys == key)[0][0])

         try:
            possible_indices = np.where(self.nonbonded_type_keys[:,0] == index_key[0])[0]
            self.nonbond_type_dict_index.append(possible_indices[int(np.where(self.nonbonded_type_keys[possible_indices,1] == index_key[1])[-1])])
         except:
            np.where(self.nonbonded_type_keys[:,0] == index_key[0])


      self.nonbond_dict_index = torch.asarray(self.nonbond_dict_index).to(self.device)

      self.nonbond_type_dict_index = torch.from_numpy(np.array(self.nonbond_type_dict_index)).to(self.device)

      self.nonbonded_type_keys = torch.asarray(self.nonbonded_type_keys).to(self.device)

      self.nonbonded_indices = torch.from_numpy(np.array(self.nonbonded_indices)).to(self.device)


      self.bead_radii = torch.nn.Parameter(torch.Tensor([1.200000e-01 for i in range(0,self.dataset['bead_types'].max()+1)]).to(self.device))

      self.bead_types = torch.nn.Parameter(torch.Tensor(self.dataset['bead_types']).to(self.device))
      

      self.dispertion_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.300000e-00  ).to(self.device))
      # self.lj_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.500000e-01  ).to(self.device))

      # self.bond_H_lj_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.500000e-01 ).to(self.device))
      self.bond_H_strength_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 1.000000e-01 ).to(self.device))


      ## Charges needs to be per bead type not different for each bead ##
      self.e_0 = torch.nn.Parameter(torch.Tensor([8.8541878128e-3]).to(self.device))
      self.e_r = torch.nn.Parameter(torch.Tensor([1]).to(self.device)) #########################[8.8541878128e-3]
      #########################################################################################################
      self.f_0 = torch.nn.Parameter(torch.Tensor([138.935458]).to(self.device))
      self.bead_charges_vals = torch.nn.Parameter(torch.reshape(torch.Tensor([i for i in conf_bead_charges.values()]).float().to(self.device), (-1,)))
      self.bead_charges_keys = np.asanyarray([i for i in conf_bead_charges.keys()])

      self.bead_charge_indices = []
      for bead_name in dataset['bead_idnames']:
         self.bead_charge_indices.append(np.where(self.bead_charges_keys == bead_name)[0])
      self.bead_charge_indices = torch.asarray(self.bead_charge_indices).to(self.device)

      # bead_charges = dataset['bead_charges']
      # self.bead_charges = torch.nn.Parameter(torch.from_numpy(bead_charges).float().to(self.device))
      # self.bead_charge = torch.multiply(bead_charges, bead_charges.T).to(self.device)[self.nonbonded_indices[0],self.nonbonded_indices[1]] # may be paramater
      
      # Bonded Interections 
      self.bond_indices = torch.from_numpy(bond_indices).to(self.device)
      
      self.equ_val_bond_dist_keys = np.asanyarray([i for i in conf_bonds.keys()])
      self.equ_val_bond_dist_vals = torch.Tensor([i for i in conf_bonds.values()]).to(self.device)
      self.spring_constant_vals = torch.nn.Parameter(torch.Tensor([4000  for i in conf_bonds.values()]).to(self.device)) 

      self.bond_dist_index = []
      for row in self.bond_indices:
         if str(self.dataset["bead_idnames"][row[0]] + '-' + self.dataset["bead_idnames"][row[1]]) in self.equ_val_bond_dist_keys:
            key = str(self.dataset["bead_idnames"][row[0]] + '-' + self.dataset["bead_idnames"][row[1]]) 
         elif str(self.dataset["bead_idnames"][row[1]] + '-' + self.dataset["bead_idnames"][row[0]]) in self.equ_val_bond_dist_keys:
            key = str(self.dataset["bead_idnames"][row[1]] + '-' + self.dataset["bead_idnames"][row[0]]) 

         self.bond_dist_index.append(np.where(self.equ_val_bond_dist_keys == key)[0][0])
      self.bond_dist_index = torch.asarray(self.bond_dist_index).to(self.device)

      # Angle Interactions
      self.angle_indices = torch.from_numpy(angle_indices).to(self.device)

      self.equ_val_angles_keys = np.asanyarray([i for i in conf_angles.keys()])
      self.equ_val_angles_vals = torch.Tensor([i for i in conf_angles.values()]).to(self.device)
      self.angle_spring_constant_vals = torch.nn.Parameter(torch.Tensor([41  for i in conf_angles.values()]).to(self.device))

      self.angle_rad_index = []

      for triple in self.angle_indices:
         if str(self.dataset["bead_idnames"][triple[0]] + '-' + self.dataset["bead_idnames"][triple[1]] + '-' + self.dataset["bead_idnames"][triple[2]]) in self.equ_val_angles_keys:
            key = str(self.dataset["bead_idnames"][triple[0]] + '-' + self.dataset["bead_idnames"][triple[1]] + '-' + self.dataset["bead_idnames"][triple[2]])
         elif str(self.dataset["bead_idnames"][triple[2]] + '-' + self.dataset["bead_idnames"][triple[1]] + '-' + self.dataset["bead_idnames"][triple[0]]) in self.equ_val_angles_keys:
            key = str(self.dataset["bead_idnames"][triple[2]] + '-' + self.dataset["bead_idnames"][triple[1]] + '-' + self.dataset["bead_idnames"][triple[0]])

         self.angle_rad_index.append(np.where(self.equ_val_angles_keys == key)[0][0])
         
      self.angle_rad_index = torch.asarray(self.angle_rad_index).to(self.device)


      # Dihedral Interactions
      self.improper_dih_indices = torch.from_numpy(improper_dih_indices).to(self.device) 

      self.equ_val_dihedrals_keys = np.asanyarray([i for i in conf_dihedrals.keys()])
      self.equ_val_dihedrals_vals = torch.Tensor([i for i in conf_dihedrals.values()]).to(self.device)
      self.dihedral_const_vals =  torch.nn.Parameter(torch.Tensor([100  for i in conf_dihedrals.values()]).to(self.device))

      self.dih_rad_index = []
      for quadruple in self.improper_dih_indices:
         self.dih_rad_index.append(np.where(self.equ_val_dihedrals_keys == self.dataset["bead_idnames"][quadruple[0]] + '-' + self.dataset["bead_idnames"][quadruple[1]] + '-' + self.dataset["bead_idnames"][quadruple[2]] + '-' + self.dataset["bead_idnames"][quadruple[3]] )[0][0])
      self.dih_rad_index = torch.asarray(self.dih_rad_index).to(self.device)

      #Water Potentials
      self.water_interactions = torch.ones(len(dataset['bead_types'])).to(self.device)

      #Proper Dihedrals
      propers=[]
      bead_idnames = []
      for index in range(0,len(dataset['bead_idnames'])):
         bead_name = dataset['bead_idnames'][index]
         bead_type, bead_function = bead_name.split("_")
         if bead_function in ['BB', 'SC1']:
            propers.append(index)
            bead_idnames.append(bead_function)
      proper_indices = []
      for index in range(0,len(propers) - 3):
         if bead_idnames[index+1] == 'SC1' and bead_idnames[index] == 'BB' and bead_idnames[index+2] == 'BB' and bead_idnames[index+3] == 'SC1':  
            proper_indices.append(np.array([propers[index+1],propers[index],propers[index+2],propers[index+3]]))

      self.proper_indices = torch.asarray(proper_indices)
      self.proper_dih_const = torch.nn.Parameter(torch.Tensor([0]).float().to(self.device))
      self.proper_phase_shift = torch.nn.Parameter(torch.Tensor(torch.zeros(len(self.proper_indices))).float().to(self.device))
      self.proper_shift = torch.nn.Parameter(torch.Tensor(torch.ones(len(self.proper_indices))).float().to(self.device))



      # Proper BB indices

      proper_BB_indices = []
      BB_indices = []
      for bead_index, bead_name in enumerate(dataset["bead_idnames"]):
         if bead_name.split('_')[1] == "BB":
            BB_indices.append(bead_index)

      for index in range(0, len(BB_indices)-3, 2):
         proper_BB_indices.append(np.array([BB_indices[index],BB_indices[index+1],BB_indices[index+2],BB_indices[index+3]]))

      self.proper_BB_indices = torch.asarray(proper_BB_indices)
      self.proper_dih_const_BB = torch.nn.Parameter(torch.Tensor([0]).float().to(self.device))
      self.proper_phase_shift_BB = torch.nn.Parameter(torch.Tensor(torch.zeros(len(self.proper_BB_indices))).float().to(self.device))
      self.proper_shift_BB = torch.nn.Parameter(torch.Tensor(torch.ones(len(self.proper_BB_indices))).float().to(self.device))


      # Phantom Force 

      self.center_of_mass = torch.mean(torch.Tensor(self.dataset['bead_pos'][0]), dim=0)

      self.phantom_distances = torch.Tensor(self.dataset['bead_pos'][0]) - self.center_of_mass
      self.phantom_norm = torch.norm(self.phantom_distances, dim=-1)

      self.phantom_force_coeff = torch.nn.Parameter(torch.Tensor(torch.zeros(len(self.dataset['bead_types']))).float().to(self.device))
      self.phantom_distance = torch.nn.Parameter(torch.Tensor(self.phantom_norm).float().to(self.device))



   def get_bead_pair_radii(self):

      selection = self.nonbonded_type_keys[self.nonbond_type_dict_index]

      bead_pair_radii = self.bead_radii[selection[:,0]] + self.bead_radii[selection[:,1]]
      
      return bead_pair_radii

      
   def phantomForces(self, bead_pos):

      center_of_mass = torch.mean(bead_pos, dim=0)

      phantom_distances = bead_pos[:,:,None] - center_of_mass
      phantom_norm = torch.norm(phantom_distances, dim=-1)
     
      phantom_energy = 0.5 * self.phantom_force_coeff[:,None] * torch.pow(phantom_norm - self.phantom_distance[:,None], 2)
      return phantom_energy







   def properDih(self, bead_pos):

      # vectors1 = bead_pos[:,self.proper_indices[:,0]] - bead_pos[:,self.proper_indices[:,1]]
      # vectors2 = bead_pos[:,self.proper_indices[:,2]] - bead_pos[:,self.proper_indices[:,1]]
      # vectors3 = bead_pos[:,self.proper_indices[:,1]] - bead_pos[:,self.proper_indices[:,2]]
      # vectors4 = bead_pos[:,self.proper_indices[:,3]] - bead_pos[:,self.proper_indices[:,2]]

      # unit_vec1 = vectors1[:] / torch.norm(vectors1, dim=-1)[:,:,None]
      # unit_vec2 = vectors2[:] / torch.norm(vectors2, dim=-1)[:,:,None]
      # unit_vec3 = vectors3[:] / torch.norm(vectors3, dim=-1)[:,:,None]
      # unit_vec4 = vectors4[:] / torch.norm(vectors4, dim=-1)[:,:,None]

      # binorm1 = torch.cross(unit_vec1, unit_vec2, dim=-1) 
      # binorm2 = torch.cross(unit_vec3, unit_vec4, dim=-1) 

      # torsion = torch.arccos(torch.sum(binorm1*binorm2, dim=-1))

      torsion = self.get_dihedrals_torch(bead_pos,self.proper_indices)


      proper_dih_energy =  torch.abs(self.proper_dih_const) * (1 + torch.cos(self.proper_shift * torsion - self.proper_phase_shift))

      return proper_dih_energy
   
   def properDihBB(self, bead_pos):

      # vectors1 = bead_pos[:,self.proper_BB_indices[:,0]] - bead_pos[:,self.proper_BB_indices[:,1]]
      # vectors2 = bead_pos[:,self.proper_BB_indices[:,2]] - bead_pos[:,self.proper_BB_indices[:,1]]
      # vectors3 = bead_pos[:,self.proper_BB_indices[:,1]] - bead_pos[:,self.proper_BB_indices[:,2]]
      # vectors4 = bead_pos[:,self.proper_BB_indices[:,3]] - bead_pos[:,self.proper_BB_indices[:,2]]

      # unit_vec1 = vectors1[:] / torch.norm(vectors1, dim=-1)[:,:,None]
      # unit_vec2 = vectors2[:] / torch.norm(vectors2, dim=-1)[:,:,None]
      # unit_vec3 = vectors3[:] / torch.norm(vectors3, dim=-1)[:,:,None]
      # unit_vec4 = vectors4[:] / torch.norm(vectors4, dim=-1)[:,:,None]

      # binorm1 = torch.cross(unit_vec1, unit_vec2, dim=-1) 
      # binorm2 = torch.cross(unit_vec3, unit_vec4, dim=-1) 

      # torsion = torch.arccos(torch.sum(binorm1*binorm2, dim=-1))

      torsion = self.get_dihedrals_torch(bead_pos,self.proper_BB_indices)

      proper_dih_energy = torch.abs(self.proper_dih_const_BB) * (1 + torch.cos(self.proper_shift_BB * torsion - self.proper_phase_shift_BB ))

      return proper_dih_energy



      

   def dihedral(self, bead_pos):

      # vectors1 = bead_pos[:,self.improper_dih_indices[:,0]] - bead_pos[:,self.improper_dih_indices[:,1]]
      # vectors2 = bead_pos[:,self.improper_dih_indices[:,2]] - bead_pos[:,self.improper_dih_indices[:,1]]
      # vectors3 = bead_pos[:,self.improper_dih_indices[:,1]] - bead_pos[:,self.improper_dih_indices[:,2]]
      # vectors4 = bead_pos[:,self.improper_dih_indices[:,3]] - bead_pos[:,self.improper_dih_indices[:,2]]

      # unit_vec1 = vectors1[:] / torch.norm(vectors1, dim=-1)[:,:,None]
      # unit_vec2 = vectors2[:] / torch.norm(vectors2, dim=-1)[:,:,None]
      # unit_vec3 = vectors3[:] / torch.norm(vectors3, dim=-1)[:,:,None]
      # unit_vec4 = vectors4[:] / torch.norm(vectors4, dim=-1)[:,:,None]

      # binorm1 = torch.cross(unit_vec1, unit_vec2, dim=-1) 
      # binorm2 = torch.cross(unit_vec3, unit_vec4, dim=-1) 

      torsion = self.get_dihedrals_torch(bead_pos,self.improper_dih_indices)- torch.pi
      
      dihedral_energy = 0.5 * torch.abs(self.dihedral_const_vals[self.dih_rad_index]) * torch.pow((torsion - self.equ_val_dihedrals_vals[self.dih_rad_index]), 2)

      return dihedral_energy
   
   def get_dihedrals_torch(self, pos: torch.Tensor, dihedral_idcs: torch.Tensor) -> torch.Tensor:
      """ Compute dihedral values (in radiants) over specified dihedral_idcs for every frame in the batch
   
         :param pos:        torch.Tensor | shape (batch, n_atoms, xyz)
         :param dihedral_idcs: torch.Tensor | shape (n_dihedrals, 4)
         :return:           torch.Tensor | shape (batch, n_dihedrals)
      """
   
      if len(pos.shape) == 2:
         pos = torch.unsqueeze(pos, dim=0)
      p = pos[:, dihedral_idcs, :]
      p0 = p[..., 0, :]
      p1 = p[..., 1, :]
      p2 = p[..., 2, :]
      p3 = p[..., 3, :]
   
      b0 = -1.0*(p1 - p0)
      b1 = p2 - p1
      b2 = p3 - p2
   
      b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True)
   
      v = b0 - torch.einsum("ijk,ikj->ij", b0, torch.transpose(b1, 1, 2))[..., None] * b1
      w = b2 - torch.einsum("ijk,ikj->ij", b2, torch.transpose(b1, 1, 2))[..., None] * b1
   
      x = torch.einsum("ijk,ikj->ij", v, torch.transpose(w, 1, 2))
      y = torch.einsum("ijk,ikj->ij", torch.cross(b1, v), torch.transpose(w, 1, 2))
   
      return torch.atan2(y, x).reshape(-1, dihedral_idcs.shape[0])

      
   
   
   def angles(self, bead_pos):
      vectors1 = bead_pos[:,self.angle_indices[:,0]] - bead_pos[:,self.angle_indices[:,1]]
      vectors2 = bead_pos[:,self.angle_indices[:,2]] - bead_pos[:,self.angle_indices[:,1]]

      unit_vec1 = vectors1[:] / torch.norm(vectors1, dim=-1)[:,:,None]
      unit_vec2 = vectors2[:] / torch.norm(vectors2, dim=-1)[:,:,None]
      
      angles = torch.arccos(torch.sum(unit_vec1 * unit_vec2, dim=-1))
      # print(self.average_angles_rad, angles)
      
      angle_energy = 0.5 * torch.abs(self.angle_spring_constant_vals[self.angle_rad_index]) * torch.pow(angles - self.equ_val_angles_vals[self.angle_rad_index], 2)
      return angle_energy
   
   def bonds(self, bead_pos):
      bond_pos = bead_pos[:,self.bond_indices,:]
      bond_distances = bond_pos[:,:,1,:]-bond_pos[:,:,0,:]
      # distances = bead_pos[:,:, None, :] - bead_pos[:,None, :, :]
      bond_norm = torch.norm(bond_distances, dim=-1)
      # triangular = torch.tril(norm, diagonal=-1)
      # bond_distances = []
      # for row in self.bond_indices:
      #       bond_distances.append(norm[tuple(row)])

               
      # bond_distances = torch.FloatTensor(bond_distances)
      bond_energy = 0.5 * torch.abs(self.spring_constant_vals[self.bond_dist_index]) * torch.pow(bond_norm - self.equ_val_bond_dist_vals[self.bond_dist_index], 2) # fix dimention
      return bond_energy
   
   # def waterPot(self, bead_pos):

   def getForceField(self):
      force_field = {}
      for i in self.state_dict().keys():
         if i in ['dispertion_const', 'lj_const']:
            force_field[i]= dict(zip(self.nonbonded_keys, self.state_dict()[i]))

         elif i in ['equ_val_bond_dist_vals', 'spring_constant_vals']:
           force_field[i]= dict(zip(self.equ_val_bond_dist_keys, self.state_dict()[i]))

         elif i in ['equ_val_angles_vals', 'angle_spring_constant_vals']:
           force_field[i]= dict(zip(self.equ_val_angles_keys, self.state_dict()[i]))

         elif i in ['dihedral_const_vals']:
           force_field[i]= dict(zip(self.equ_val_dihedrals_keys, self.state_dict()[i]))

         elif i in ['bead_charges_vals']:
           force_field[i]= list(zip(self.bead_charges_keys, self.state_dict()[i]))

      return force_field

      



   """A central harmonic potential as a static compute graph"""
   def forward(self, positions: torch.Tensor):
         """The forward method returns the energy computed from positions.

         Parameters
         ----------
         positions : torch.Tensor with shape (nparticles,3)
            positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

         Returns
         -------
         potential : torch.Scalar
            The potential energy (in kJ/mol)
         """
         #TODO: DO energy minimization from chignolin, simulate, and return potential as a tensor
         
         # convert as if it has batches
         if len(positions.shape) == 2:
            positions = positions[None,:,:]

         bead_pos = positions.to(self.device)
            
            
         # else:   
         #    print(positions.shape)
         #    # positions = positions[None,:,:]
         
         #    print("batches = ", withBatches)
         
         #    # for bead_pos in bead_poses:
         #    bead_pos = positions.to(self.device)

            # Calculate any function on the position, like norm of the distance, and keep the forces array shape
            # The distances would be between beads in the same frame, not between frames
         

         #We need to compute the energy only between beads that are bonded, thus we need the 'topology' (random bonds created above)
         #We want to learn the equilibrium distance just by looking at the dataset
         # energy.append(intermediate_energy.sum()+angle_energy.sum())
        
         # Update energy_contribution field of dataset as a torch tensor
         # self.dataset['energy_contribution'] = np.asanyarray(energy)




         # Calculate any function on the position, like norm of the distance, and keep the forces array shape
         # The distances would be between beads in the same frame, not between frames

         distances = bead_pos[:,:, None, :] - bead_pos[:,None, :, :]
         distances2 = distances[:,self.nonbonded_indices[0],self.nonbonded_indices[1],:]
         norm = torch.norm(distances2, dim=-1)

         bead_pair_radius = self.get_bead_pair_radii()


         lj_energy = 4 * torch.multiply(torch.abs(self.dispertion_const[self.nonbond_dict_index]),(torch.pow(torch.div(bead_pair_radius, norm), 12)
                                                                                                    - torch.pow(torch.div(bead_pair_radius, norm), 6)))

         Hbond_lj_energy = torch.multiply(torch.abs(self.bond_H_strength_const[self.nonbond_dict_index]),(torch.pow(torch.div(bead_pair_radius, norm), 6
                                                                                                                    ) - torch.pow(torch.div(bead_pair_radius, norm), 4)))


         bead_charge = torch.multiply(self.bead_charges_vals[self.bead_charge_indices], self.bead_charges_vals[self.bead_charge_indices].mT)[self.nonbonded_indices[0],self.nonbonded_indices[1]]
         coulumb_energy = torch.div(bead_charge, norm) * self.f_0 / self.e_r #/ (4 * self.e_0 * torch.pi) #


         bond_energy = self.bonds(bead_pos)

         

         angle_energy = self.angles(bead_pos)

         dihedral_energy = self.dihedral(bead_pos)

         proper_dih_energy = self.properDih(bead_pos)

         proper_dih_energy_BB = self.properDihBB(bead_pos) 

         # phantom_energy = self.phantomForces(bead_pos)

        #  force_field = dict(zip(self.nonbonded_keys, self.dispertion_const))
         
         energies = torch.sum(bond_energy)  + torch.sum(coulumb_energy) + torch.sum(dihedral_energy)  + torch.sum(proper_dih_energy_BB) + torch.sum(lj_energy) +torch.sum(Hbond_lj_energy) + torch.sum(angle_energy) #+ torch.sum(proper_dih_energy)  # + torch.sum(phantom_energy)  

  
         return energies
   
class ForceMapper(torch.nn.Module):
   def __init__(self, module: torch.nn.Module) -> None:
         super(ForceMapper, self).__init__()
         self.module = module
 
   def forward(self, position: torch.Tensor, train_mode = False) -> dict:

         # old_requires_grad: List[bool] = []
            # old_requires_grad.append(data[k].requires_grad)
         if train_mode == True:
            multip_unit = 1
         else:
            multip_unit = 10
         position.requires_grad_(True)
         position = position
         # total_energy = torch.tensor(0, dtype=torch.float32, device=self.parameters()[0].device())
         # run model
         
         energy = self.module(position)
         total_energy = energy 
               
         # Get grads
         grads = torch.autograd.grad(
            # TODO:
            # This makes sense for scalar batch-level or batch-wise outputs, specifically because d(sum(batches))/d wrt = sum(d batch / d wrt) = d my_batch / d wrt
            # for a well-behaved example level like energy where d other_batch / d wrt is always zero. (In other words, the energy of example 1 in the batch is completely unaffect by changes in the position of atoms in another example.)
            # This should work for any gradient of energy, but could act suspiciously and unexpectedly for arbitrary gradient outputs, if they ever come up
            total_energy,
            [position],
            create_graph=self.training,  # needed to allow gradients of this output during training
        )

         forces = -grads[0] 
         
         return forces
    

 