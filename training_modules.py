import torch
import numpy as np
from torch import Tensor

class ForceModel(torch.nn.Module):
   def __init__(self, dataset, conf_bonds: dict, conf_angles: dict, conf_dihedrals: dict, conf_bead_charges: dict):
      super().__init__()
      pos2unit = 1.
      eng2unit = 0.239006
      pos2unit = 1.
      eng2unit = 0.239006

      # Define Indices
      bond_indices:         np.ndarray = dataset['bond_indices']
      angle_indices:        np.ndarray = dataset['angle_indices']
      improper_dih_indices: np.ndarray = dataset['improper_dih_indices']

      bond_indices = np.unique(np.sort(bond_indices, axis=1), axis=0) # Safety measure

      self.h_bond_selection = torch.Tensor([15,16]).int().to(self.device)

      #
      bead_types = dataset['bead_types']
      num_bead_types = len(bead_types)
      
      # Remove bonded indices form nonbonded
      # upper triangluar transform index finder given the indices of the bead first is smaller than the second bead index
      
      all_indices = np.triu_indices(num_bead_types, 1)
      first_bead  = bond_indices[:, 0]
      second_bead = bond_indices[:, 1]

      first_bead  = np.append(first_bead, angle_indices[:,0])
      second_bead = np.append(second_bead, angle_indices[:,2])

      index = ((first_bead[:] * (first_bead[:] + 1)) / 2 + first_bead[:] * (num_bead_types - first_bead[:] - 1) + second_bead[:] - first_bead[:] - 1).astype(int)
      self.nonbonded_indices = [[],[]]
      self.nonbonded_indices[0] = np.delete(all_indices[0],index,None)
      self.nonbonded_indices[1] = np.delete(all_indices[1],index,None)
      
      # NonBonded Interactions
      bead_idnames = dataset['bead_idnames']

      nonbonded_names = [
         [
            bead_idnames[self.nonbonded_indices[0][i]],
            bead_idnames[self.nonbonded_indices[1][i]],
         ]
         for i in range(len(self.nonbonded_indices[0]))
      ]
      nonbonded_names = np.array(nonbonded_names)
      self.nonbonded_dict = {}
      for i, j in nonbonded_names:
         if (str(i + "-" + j) and str(j + "-" + i)) not in self.nonbonded_dict.keys():
            self.nonbonded_dict[i + "-" + j] = 1

      self.nonbonded_keys = np.asanyarray([i for i in self.nonbonded_dict.keys()])
      self.nonbonded_vals = torch.from_numpy(np.asanyarray([i for i in self.nonbonded_dict.values()])).to(self.device)

      self.nonbonded_type_keys = np.unique(np.sort(np.array([[bead_types[self.nonbonded_indices[0][i]],bead_types[self.nonbonded_indices[1][i]]] for i in range(len(self.nonbonded_indices[0]))]), axis=1), axis=0)

      self.nonbond_dict_index = []
      nonbonded_pairs = [[self.nonbonded_indices[0][i],self.nonbonded_indices[1][i]] for i in range(len(self.nonbonded_indices[0]))]
      self.nonbond_type_dict_index = []
      #self.nonbonded_pairs = [[self.nonbonded_indices[0][i],self.nonbonded_indices[1][i]] for i in range(len(self.nonbonded_indices[0]))]

      for pair in nonbonded_pairs:
         if str(bead_idnames[pair[0]] + '-' + bead_idnames[pair[1]]) in self.nonbonded_keys:
            key = str(bead_idnames[pair[0]] + '-' + bead_idnames[pair[1]]) 
         elif str(bead_idnames[pair[1]] + '-' + bead_idnames[pair[0]]) in self.nonbonded_keys:
            key = str(bead_idnames[pair[1]] + '-' + bead_idnames[pair[0]])

         if any([all([bead_types[pair[0]], bead_types[pair[1]]] == arr) for arr in self.nonbonded_type_keys]):
            index_key = np.array([bead_types[pair[0]], bead_types[pair[1]]])
         elif any([all([bead_types[pair[1]], bead_types[pair[0]]] == arr) for arr in self.nonbonded_type_keys]):
            index_key = np.array([bead_types[pair[1]], bead_types[pair[0]]])

         self.nonbond_dict_index.append(np.where(self.nonbonded_keys == key)[0][0])

         try:
            possible_indices = np.where(self.nonbonded_type_keys[:,0] == index_key[0])[0]
            self.nonbond_type_dict_index.append(possible_indices[int(np.where(self.nonbonded_type_keys[possible_indices,1] == index_key[1])[-1])])
         except:
            np.where(self.nonbonded_type_keys[:,0] == index_key[0])


         try:
            possible_indices = np.where(self.nonbonded_type_keys[:,0] == index_key[0])[0]
            self.nonbond_type_dict_index.append(possible_indices[int(np.where(self.nonbonded_type_keys[possible_indices,1] == index_key[1])[-1])])
         except:
            np.where(self.nonbonded_type_keys[:,0] == index_key[0])


      self.nonbond_dict_index = torch.asarray(self.nonbond_dict_index)

      self.nonbond_type_dict_index = torch.from_numpy(np.array(self.nonbond_type_dict_index)).to(self.device)

      self.nonbonded_type_keys = torch.asarray(self.nonbonded_type_keys).to(self.device)

      self.nonbonded_indices = torch.from_numpy(np.array(self.nonbonded_indices))


      self.bead_radii = torch.nn.Parameter(torch.Tensor([1.200000e-01 for i in range(0,bead_types.max()+1)]).to(self.device))

      self.bead_types = torch.nn.Parameter(torch.Tensor(bead_types).to(self.device))
      

      self.dispertion_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.300000e-00  ).to(self.device))
      # self.lj_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.500000e-01  ).to(self.device))

      # self.bond_H_lj_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.500000e-01 ).to(self.device))
      self.bond_H_strength_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 1.000000e-01 ).to(self.device))


      ## Charges needs to be per bead type not different for each bead ##
      self.e_0 = torch.nn.Parameter(torch.Tensor([8.8541878128e-3]))
      self.e_r = torch.nn.Parameter(torch.Tensor([1])) #########################[8.8541878128e-3]
      self.e_0 = torch.nn.Parameter(torch.Tensor([8.8541878128e-3]))
      self.e_r = torch.nn.Parameter(torch.Tensor([1])) #########################[8.8541878128e-3]
      #########################################################################################################
      self.f_0 = torch.nn.Parameter(torch.Tensor([138.935458]))
      self.bead_charges_vals = torch.nn.Parameter(torch.reshape(torch.Tensor([i for i in conf_bead_charges.values()]).float(), (-1,)))
      self.f_0 = torch.nn.Parameter(torch.Tensor([138.935458]))
      self.bead_charges_vals = torch.nn.Parameter(torch.reshape(torch.Tensor([i for i in conf_bead_charges.values()]).float(), (-1,)))
      self.bead_charges_keys = np.asanyarray([i for i in conf_bead_charges.keys()])

      self.bead_charge_indices = []
      for bead_name in bead_idnames:
         self.bead_charge_indices.extend(np.where(self.bead_charges_keys == bead_name)[0])
      self.bead_charge_indices = torch.asarray(self.bead_charge_indices)
      self.bead_charge_indices = torch.asarray(self.bead_charge_indices)

      # bead_charges = dataset['bead_charges']
      # self.bead_charges = torch.nn.Parameter(torch.from_numpy(bead_charges).float())
      # self.bead_charge = torch.multiply(bead_charges, bead_charges.T)[self.nonbonded_indices[0],self.nonbonded_indices[1]] # may be paramater
      # self.bead_charges = torch.nn.Parameter(torch.from_numpy(bead_charges).float())
      # self.bead_charge = torch.multiply(bead_charges, bead_charges.T)[self.nonbonded_indices[0],self.nonbonded_indices[1]] # may be paramater
      
      # Bonded Interections 
      self.bond_indices = torch.from_numpy(bond_indices)
      self.bond_indices = torch.from_numpy(bond_indices)
      
      self.equ_val_bond_dist_keys = np.asanyarray([i for i in conf_bonds.keys()])
      self.equ_val_bond_dist_vals = torch.nn.Parameter(torch.Tensor([i for i in conf_bonds.values()]) * pos2unit)
      self.spring_constant_vals = torch.nn.Parameter(torch.Tensor([1000  for i in conf_bonds.values()]) * eng2unit)
      self.equ_val_bond_dist_vals = torch.nn.Parameter(torch.Tensor([i for i in conf_bonds.values()]) * pos2unit)
      self.spring_constant_vals = torch.nn.Parameter(torch.Tensor([1000  for i in conf_bonds.values()]) * eng2unit)

      self.bond_dist_index = []
      for row in self.bond_indices:
         if str(bead_idnames[row[0]] + '-' + bead_idnames[row[1]]) in self.equ_val_bond_dist_keys:
            key = str(bead_idnames[row[0]] + '-' + bead_idnames[row[1]]) 
         elif str(bead_idnames[row[1]] + '-' + bead_idnames[row[0]]) in self.equ_val_bond_dist_keys:
            key = str(bead_idnames[row[1]] + '-' + bead_idnames[row[0]]) 

         self.bond_dist_index.append(np.where(self.equ_val_bond_dist_keys == key)[0][0])
      self.bond_dist_index = torch.asarray(self.bond_dist_index)
      self.bond_dist_index = torch.asarray(self.bond_dist_index)

      # Angle Interactions
      self.angle_indices = torch.from_numpy(angle_indices)
      self.angle_indices = torch.from_numpy(angle_indices)

      self.equ_val_angles_keys = np.asanyarray([i for i in conf_angles.keys()])
      self.equ_val_angles_vals = torch.nn.Parameter(torch.Tensor([i for i in conf_angles.values()]))
      self.angle_spring_constant_vals = torch.nn.Parameter(torch.Tensor([10  for i in conf_angles.values()]) * eng2unit)
      self.equ_val_angles_vals = torch.nn.Parameter(torch.Tensor([i for i in conf_angles.values()]))
      self.angle_spring_constant_vals = torch.nn.Parameter(torch.Tensor([10  for i in conf_angles.values()]) * eng2unit)

      self.angle_rad_index = []

      for triple in self.angle_indices:
         if str(bead_idnames[triple[0]] + '-' + bead_idnames[triple[1]] + '-' + bead_idnames[triple[2]]) in self.equ_val_angles_keys:
            key = str(bead_idnames[triple[0]] + '-' + bead_idnames[triple[1]] + '-' + bead_idnames[triple[2]])
         elif str(bead_idnames[triple[2]] + '-' + bead_idnames[triple[1]] + '-' + bead_idnames[triple[0]]) in self.equ_val_angles_keys:
            key = str(bead_idnames[triple[2]] + '-' + bead_idnames[triple[1]] + '-' + bead_idnames[triple[0]])

         self.angle_rad_index.append(np.where(self.equ_val_angles_keys == key)[0][0])
         
      self.angle_rad_index = torch.asarray(self.angle_rad_index)
      self.angle_rad_index = torch.asarray(self.angle_rad_index)

      # Dihedral Interactions
      self.improper_dih_indices = torch.from_numpy(improper_dih_indices)
      self.improper_dih_indices = torch.from_numpy(improper_dih_indices)

      self.equ_val_dihedrals_keys = np.asanyarray([i for i in conf_dihedrals.keys()])
      self.equ_val_dihedrals_vals = torch.nn.Parameter(torch.Tensor([i for i in conf_dihedrals.values()]))
      self.dihedral_const_vals =  torch.nn.Parameter(torch.Tensor([1  for i in conf_dihedrals.values()]) * eng2unit)
      self.equ_val_dihedrals_vals = torch.nn.Parameter(torch.Tensor([i for i in conf_dihedrals.values()]))
      self.dihedral_const_vals =  torch.nn.Parameter(torch.Tensor([1  for i in conf_dihedrals.values()]) * eng2unit)

      self.dih_rad_index = []
      for quadruple in self.improper_dih_indices:
         self.dih_rad_index.append(np.where(self.equ_val_dihedrals_keys == bead_idnames[quadruple[0]] + '-' + bead_idnames[quadruple[1]] + '-' + bead_idnames[quadruple[2]] + '-' + bead_idnames[quadruple[3]] )[0][0])
      self.dih_rad_index = torch.asarray(self.dih_rad_index)
      self.dih_rad_index = torch.asarray(self.dih_rad_index)

      # Water Potentials
      self.water_interactions = torch.ones(len(dataset['bead_types']))
      # Water Potentials
      self.water_interactions = torch.ones(len(dataset['bead_types']))

      # Proper Dihedrals
      # Proper Dihedrals
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
      self.proper_dih_const = torch.nn.Parameter(torch.Tensor([50 for _ in range(0,len(self.proper_indices))]).float())
      self.proper_phase_shift = torch.nn.Parameter(torch.Tensor(torch.zeros(len(self.proper_indices))).float())
      self.proper_shift = torch.nn.Parameter(torch.Tensor(torch.ones(len(self.proper_indices))).float())

      # Proper BB indices !!! We take one torsion and skip 2 !!!

      proper_BB_indices = []
      BB_indices = []
      for bead_index, bead_name in enumerate(dataset["bead_idnames"]):
         if bead_name.split('_')[1] == "BB":
            BB_indices.append(bead_index)

      for index in range(0, len(BB_indices)-3, 2):
         proper_BB_indices.append(np.array([BB_indices[index],BB_indices[index+1],BB_indices[index+2],BB_indices[index+3]]))

      self.proper_BB_indices = torch.asarray(proper_BB_indices)
      self.proper_dih_const_BB = torch.nn.Parameter(torch.Tensor([50 for i in range(0,len(self.proper_BB_indices))]).float())
      self.proper_phase_shift_BB = torch.nn.Parameter(torch.Tensor(torch.zeros(len(self.proper_BB_indices))).float())
      self.proper_shift_BB = torch.nn.Parameter(torch.Tensor(torch.ones(len(self.proper_BB_indices))).float())

   def bonds(self, bead_pos):
      bond_pos = bead_pos[:, self.bond_indices, :]
      bond_distances = bond_pos[..., 1, :]-bond_pos[..., 0, :]
      bond_norm = torch.norm(bond_distances, dim=-1)
      
      bond_energy = (
         0.5 * torch.abs(self.spring_constant_vals[self.bond_dist_index]) *
         torch.max(
            torch.zeros_like(bond_norm),
            torch.pow(bond_norm - self.equ_val_bond_dist_vals[self.bond_dist_index], 2) - 0.005
         )
         torch.max(
            torch.zeros_like(bond_norm),
            torch.pow(bond_norm - self.equ_val_bond_dist_vals[self.bond_dist_index], 2) - 0.005
         )
      )
      return bond_energy
      
   def angles(self, bead_pos):
      angles = get_angles(bead_pos, self.angle_indices)
      
      angle_energy = (
         0.5 * torch.abs(self.angle_spring_constant_vals[self.angle_rad_index]) *
         torch.max(
            torch.zeros_like(angles),
            torch.pow(angles - self.equ_val_angles_vals[self.angle_rad_index], 2) - 0.05
         )
         torch.max(
            torch.zeros_like(angles),
            torch.pow(angles - self.equ_val_angles_vals[self.angle_rad_index], 2) - 0.05
         )
      )

      return angle_energy
   
   def dihedrals(self, bead_pos):
      torsion = get_dihedrals(bead_pos, self.improper_dih_indices) 

      dihedral_energy = (
         0.5 * torch.abs(self.dihedral_const_vals[self.dih_rad_index]) *
         torch.max(
            torch.zeros_like(torsion),
            torch.pow((torsion - self.equ_val_dihedrals_vals[self.dih_rad_index]), 2) - 0.05
         )
         torch.max(
            torch.zeros_like(torsion),
            torch.pow((torsion - self.equ_val_dihedrals_vals[self.dih_rad_index]), 2) - 0.05
         )
      )

      return dihedral_energy

   def properDih(self, bead_pos):
      torsion = get_dihedrals(bead_pos, self.proper_indices)

      proper_dih_energy = (
         torch.abs(self.proper_dih_const) *
         torch.max(
            torch.zeros_like(torsion),
            (1 + torch.cos(self.proper_shift * torsion - self.proper_phase_shift)) - 0.05
         )
         torch.max(
            torch.zeros_like(torsion),
            (1 + torch.cos(self.proper_shift * torsion - self.proper_phase_shift)) - 0.05
         )
      )

      return proper_dih_energy
   
   def properDihBB(self, bead_pos):
      torsion = get_dihedrals(bead_pos, self.proper_BB_indices)

      proper_dih_energy = (
         torch.abs(self.proper_dih_const_BB) *
         torch.max(
            torch.zeros_like(torsion),
            (1 + torch.cos(self.proper_shift_BB * torsion - self.proper_phase_shift_BB)) - 0.05
         )
         torch.max(
            torch.zeros_like(torsion),
            (1 + torch.cos(self.proper_shift_BB * torsion - self.proper_phase_shift_BB)) - 0.05
         )
      )

      return proper_dih_energy



      

   # def dihedral(self, bead_pos):

   #    # vectors1 = bead_pos[:,self.improper_dih_indices[:,0]] - bead_pos[:,self.improper_dih_indices[:,1]]
   #    # vectors2 = bead_pos[:,self.improper_dih_indices[:,2]] - bead_pos[:,self.improper_dih_indices[:,1]]
   #    # vectors3 = bead_pos[:,self.improper_dih_indices[:,1]] - bead_pos[:,self.improper_dih_indices[:,2]]
   #    # vectors4 = bead_pos[:,self.improper_dih_indices[:,3]] - bead_pos[:,self.improper_dih_indices[:,2]]

   #    # unit_vec1 = vectors1[:] / torch.norm(vectors1, dim=-1)[:,:,None]
   #    # unit_vec2 = vectors2[:] / torch.norm(vectors2, dim=-1)[:,:,None]
   #    # unit_vec3 = vectors3[:] / torch.norm(vectors3, dim=-1)[:,:,None]
   #    # unit_vec4 = vectors4[:] / torch.norm(vectors4, dim=-1)[:,:,None]

   #    # binorm1 = torch.cross(unit_vec1, unit_vec2, dim=-1) 
   #    # binorm2 = torch.cross(unit_vec3, unit_vec4, dim=-1) 

   #    torsion = self.get_dihedrals_torch(bead_pos,self.improper_dih_indices)- torch.pi
      
   #    dihedral_energy = 0.5 * torch.abs(self.dihedral_const_vals[self.dih_rad_index]) * torch.pow((torsion - self.equ_val_dihedrals_vals[self.dih_rad_index]), 2)

   #    return dihedral_energy
   
   # def get_dihedrals_torch(self, pos: torch.Tensor, dihedral_idcs: torch.Tensor) -> torch.Tensor:
   #    """ Compute dihedral values (in radiants) over specified dihedral_idcs for every frame in the batch
   
   #       :param pos:        torch.Tensor | shape (batch, n_atoms, xyz)
   #       :param dihedral_idcs: torch.Tensor | shape (n_dihedrals, 4)
   #       :return:           torch.Tensor | shape (batch, n_dihedrals)
   #    """
   
   #    if len(pos.shape) == 2:
   #       pos = torch.unsqueeze(pos, dim=0)
   #    p = pos[:, dihedral_idcs, :]
   #    p0 = p[..., 0, :]
   #    p1 = p[..., 1, :]
   #    p2 = p[..., 2, :]
   #    p3 = p[..., 3, :]
   
   #    b0 = -1.0*(p1 - p0)
   #    b1 = p2 - p1
   #    b2 = p3 - p2
   
   #    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True)
   
   #    v = b0 - torch.einsum("ijk,ikj->ij", b0, torch.transpose(b1, 1, 2))[..., None] * b1
   #    w = b2 - torch.einsum("ijk,ikj->ij", b2, torch.transpose(b1, 1, 2))[..., None] * b1
   
   #    x = torch.einsum("ijk,ikj->ij", v, torch.transpose(w, 1, 2))
   #    y = torch.einsum("ijk,ikj->ij", torch.cross(b1, v), torch.transpose(w, 1, 2))
   
   #    return torch.atan2(y, x).reshape(-1, dihedral_idcs.shape[0])

      
   
   
   # def angles(self, bead_pos):
   #    vectors1 = bead_pos[:,self.angle_indices[:,0]] - bead_pos[:,self.angle_indices[:,1]]
   #    vectors2 = bead_pos[:,self.angle_indices[:,2]] - bead_pos[:,self.angle_indices[:,1]]

   #    unit_vec1 = vectors1[:] / torch.norm(vectors1, dim=-1)[:,:,None]
   #    unit_vec2 = vectors2[:] / torch.norm(vectors2, dim=-1)[:,:,None]
      
   #    angles = torch.arccos(torch.sum(unit_vec1 * unit_vec2, dim=-1))
   #    # print(self.average_angles_rad, angles)
      
   #    angle_energy = 0.5 * torch.abs(self.angle_spring_constant_vals[self.angle_rad_index]) * torch.pow(angles - self.equ_val_angles_vals[self.angle_rad_index], 2)
   #    return angle_energy
   
   # def bonds(self, bead_pos):
   #    bond_pos = bead_pos[:,self.bond_indices,:]
   #    bond_distances = bond_pos[:,:,1,:]-bond_pos[:,:,0,:]
   #    # distances = bead_pos[:,:, None, :] - bead_pos[:,None, :, :]
   #    bond_norm = torch.norm(bond_distances, dim=-1)
   #    # triangular = torch.tril(norm, diagonal=-1)
   #    # bond_distances = []
   #    # for row in self.bond_indices:
   #    #       bond_distances.append(norm[tuple(row)])

               
   #    # bond_distances = torch.FloatTensor(bond_distances)
   #    bond_energy = 0.5 * torch.abs(self.spring_constant_vals[self.bond_dist_index]) * torch.pow(bond_norm - self.equ_val_bond_dist_vals[self.bond_dist_index], 2) # fix dimention
   #    return bond_energy
   
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
            positions = positions[None, ...]

         bead_pos = positions
         bead_pos = positions

         all_dist = bead_pos[..., None, :] - bead_pos[:, None, ...]
         nonbonded_dist = all_dist[:, self.nonbonded_indices[0], self.nonbonded_indices[1], :]
         nonbonded_norm = torch.norm(nonbonded_dist, dim=-1)
         # cutoff_indices = torch.where(nonbonded_norm > 1.2)
         # nonbonded_norm = nonbonded_norm * 
         ###########################
         x=nonbonded_norm
         p=12
         factor = 1.0 / float(1.2)
         x = x * factor
      
         out = 1.0
         out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
         out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
         out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
         cutoff_matrix = out * (x < 1.0)
         #############################
         # cutoff_matrix = _poly_cutoff(nonbonded_norm, 1.2, p=12)
         # cutoff_matrix = torch.where(nonbonded_norm > 1.2, 0, 1.0)

         h_bond_norm = torch.norm(all_dist[:, [10,10], [19,18], :], dim=-1)


         bead_pair_radius = self.get_bead_pair_radii()

         eps             = torch.abs(self.dispertion_const[self.nonbond_dict_index]) 
         sigma_over_r    = torch.div(bead_pair_radius, nonbonded_norm)* cutoff_matrix
         lj_energy       = 4 * torch.einsum('j,ij->ij', eps, torch.pow(sigma_over_r, 12) - torch.pow(sigma_over_r, 6))

         # epsH            = torch.abs(self.bond_H_strength_const[self.nonbond_dict_index]) 
         # sigma_over_rH   = torch.div(bead_pair_radius, nonbonded_norm)* cutoff_matrix
         # Hbond_lj_energy = torch.einsum('j,ij->ij', epsH, torch.pow(sigma_over_rH, 6) - torch.pow(sigma_over_rH, 4))

         

         epsH            = torch.abs(self.bond_H_strength_const[self.h_bond_selection]) 
         sigma_over_rH   = torch.div(bead_pair_radius[self.h_bond_selection], h_bond_norm)
         Hbond_lj_energy = torch.einsum('j,ij->ij', epsH, torch.pow(sigma_over_rH, 6) - torch.pow(sigma_over_rH, 4))

         charges = self.bead_charges_vals[self.bead_charge_indices]
         bead_charge = torch.einsum('i,j->ij', charges, charges)[self.nonbonded_indices[0], self.nonbonded_indices[1]] 
         coulumb_energy = torch.div(bead_charge, nonbonded_norm) * self.f_0 / self.e_r * cutoff_matrix #/ (4 * self.e_0 * torch.pi) #

         bond_energy = self.bonds(bead_pos)

         

         angle_energy = self.angles(bead_pos)

         dihedral_energy = self.dihedrals(bead_pos)

         proper_dih_energy = self.properDih(bead_pos)

         proper_dih_energy_BB = self.properDihBB(bead_pos) 

         
         energies = (
            torch.sum(bond_energy) +
            torch.sum(angle_energy) +
            torch.sum(dihedral_energy) +
            torch.sum(proper_dih_energy_BB) +
            torch.sum(coulumb_energy) +
            torch.sum(lj_energy) +
            torch.sum(Hbond_lj_energy) +
            torch.sum(proper_dih_energy) 
         )
  
         return energies
   
class ForceMapper(torch.nn.Module):
   def __init__(self, module: torch.nn.Module) -> None:
         super(ForceMapper, self).__init__()
         self.module = module
 
   def forward(self, position: torch.Tensor) -> dict:
         position.requires_grad_(True)
         energy = self.module(position)
               
         # Get grads
         grads = torch.autograd.grad(
            # TODO:
            # This makes sense for scalar batch-level or batch-wise outputs, specifically because d(sum(batches))/d wrt = sum(d batch / d wrt) = d my_batch / d wrt
            # for a well-behaved example level like energy where d other_batch / d wrt is always zero. (In other words, the energy of example 1 in the batch is completely unaffect by changes in the position of atoms in another example.)
            # This should work for any gradient of energy, but could act suspiciously and unexpectedly for arbitrary gradient outputs, if they ever come up
            energy,
            [position],
            create_graph=self.training,  # needed to allow gradients of this output during training
        )

         forces = -grads[0] 
         
         return forces


def get_angles(pos: torch.Tensor, angle_idcs: torch.Tensor) -> torch.Tensor:
   """ Compute angle values (in radiants) over specified angle_idcs for every frame in the batch

      :param pos:        torch.Tensor | shape (batch, n_atoms, xyz)
      :param angle_idcs: torch.Tensor | shape (n_angles, 3)
      :return:           torch.Tensor | shape (batch, n_angles)
   """
   vectors1 = pos[:, angle_idcs[:,0]] - pos[:, angle_idcs[:,1]]
   vectors2 = pos[:, angle_idcs[:,2]] - pos[:, angle_idcs[:,1]]

   unit_vec1 = vectors1[:] / torch.norm(vectors1, dim=-1)[:,:,None]
   unit_vec2 = vectors2[:] / torch.norm(vectors2, dim=-1)[:,:,None]
   
   return torch.arccos(torch.sum(unit_vec1 * unit_vec2, dim=-1))


def get_dihedrals(pos: torch.Tensor, dihedral_idcs: torch.Tensor) -> torch.Tensor:
   """ Compute dihedral values (in radiants) over specified dihedral_idcs for every frame in the batch

      :param pos:           torch.Tensor | shape (batch, n_atoms, xyz)
      :param dihedral_idcs: torch.Tensor | shape (n_dihedrals, 4)
      :return:              torch.Tensor | shape (batch, n_dihedrals)
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


@torch.jit.script
def _poly_cutoff(x: Tensor, r_max: float, p: float = 6.0) -> Tensor:
    factor = 1.0 / r_max
    x_scaled = x * factor

    term1 = ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x_scaled, p)
    term2 = p * (p + 2.0) * torch.pow(x_scaled, p + 1.0)
    term3 = (p * (p + 1.0) / 2.0) * torch.pow(x_scaled, p + 2.0)

    out = 1.0 - term1 + term2 - term3

    # Apply the cutoff condition
    mask = x_scaled < 1.0
    out = out * mask

    return out