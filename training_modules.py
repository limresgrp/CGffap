import torch
import numpy as np
from torch import Tensor

class ForceModel(torch.nn.Module):
   def __init__(
         self,
         dataset,
         conf_bonds: dict,
         conf_angles: dict,
         conf_dihedrals: dict,
         conf_bead_charges: dict,
         pos2unit: float = 1.,
         eng2unit: float = 1.,
      ):
      super().__init__()

      # ----- MISC ----- #
      
      self.r_max = 10. * pos2unit

      # ----- DATASET ----- #

      bead_idnames = dataset['bead_idnames']
      bead_names   = dataset['bead_names']
      bead_types   = dataset['bead_types']
      num_beads    = len(bead_types)
      bond_indices:         np.ndarray = dataset['bond_indices']
      angle_indices:        np.ndarray = dataset['angle_indices']
      improper_dih_indices: np.ndarray = dataset['improper_dih_indices']

      # ----- BONDED ----- #

      # --- BONDS --- #

      self.bond_indices = torch.from_numpy(bond_indices)
      
      self.equ_val_bond_dist_keys = np.array(list(conf_bonds.keys()))
      self.register_buffer("equ_val_bond_dist_vals", torch.tensor(list(conf_bonds.values())) * pos2unit)
      self.spring_constant_vals   = torch.nn.Parameter(100. * torch.ones_like(self.equ_val_bond_dist_vals) * eng2unit)

      bond_dist_index = []
      for bond_ids in bond_indices:
         if str(bead_idnames[bond_ids[0]] + '-' + bead_idnames[bond_ids[1]]) in self.equ_val_bond_dist_keys:
            key = str(bead_idnames[bond_ids[0]] + '-' + bead_idnames[bond_ids[1]]) 
         elif str(bead_idnames[bond_ids[1]] + '-' + bead_idnames[bond_ids[0]]) in self.equ_val_bond_dist_keys:
            key = str(bead_idnames[bond_ids[1]] + '-' + bead_idnames[bond_ids[0]]) 

         bond_dist_index.append(np.where(self.equ_val_bond_dist_keys == key)[0][0])
      
      self.bond_dist_index = torch.tensor(bond_dist_index)

      # --- ANGLES --- #
      
      self.angle_indices = torch.from_numpy(angle_indices)

      self.equ_val_angles_keys        = np.array(list(conf_angles.keys()))
      self.register_buffer('equ_val_angles_vals', torch.tensor(list(conf_angles.values())))
      self.angle_spring_constant_vals = torch.nn.Parameter(10. * torch.ones_like(self.equ_val_angles_vals) * eng2unit)

      angle_rad_index = []
      for angle_ids in angle_indices:
         if str(bead_idnames[angle_ids[0]] + '-' + bead_idnames[angle_ids[1]] + '-' + bead_idnames[angle_ids[2]]) in self.equ_val_angles_keys:
            key = str(bead_idnames[angle_ids[0]] + '-' + bead_idnames[angle_ids[1]] + '-' + bead_idnames[angle_ids[2]])
         elif str(bead_idnames[angle_ids[2]] + '-' + bead_idnames[angle_ids[1]] + '-' + bead_idnames[angle_ids[0]]) in self.equ_val_angles_keys:
            key = str(bead_idnames[angle_ids[2]] + '-' + bead_idnames[angle_ids[1]] + '-' + bead_idnames[angle_ids[0]])

         angle_rad_index.append(np.where(self.equ_val_angles_keys == key)[0][0])
      
      self.angle_rad_index = torch.tensor(angle_rad_index)

      # --- IMPROPER DIHEDRALS --- #

      self.improper_dih_indices = torch.from_numpy(improper_dih_indices)

      self.equ_val_dihedrals_keys = np.array(list(conf_dihedrals.keys()))
      self.register_buffer("equ_val_dihedrals_vals", torch.tensor(list(conf_dihedrals.values())))
      self.dihedral_const_vals    = torch.nn.Parameter(1. * torch.ones_like(self.equ_val_dihedrals_vals) * eng2unit)

      dih_rad_index = []
      for dih_ids in self.improper_dih_indices:
         dih_rad_index.append(np.where(self.equ_val_dihedrals_keys == bead_idnames[dih_ids[0]] + '-' + bead_idnames[dih_ids[1]] + '-' + bead_idnames[dih_ids[2]] + '-' + bead_idnames[dih_ids[3]] )[0][0])
      
      self.dih_rad_index = torch.tensor(dih_rad_index)

      # --- PROPER DIHEDRALS --- #
      # TODO make proper dihedrals non-system specific as impropers

      # - SBBS - #

      sbbs_idcs = []
      sbbs_bead_names = []
      for index, bead_name in enumerate(bead_names):
         if bead_name in ['BB', 'SC1']:
            sbbs_idcs.append(index)
            sbbs_bead_names.append(bead_name)
      sbbs_idcs = np.array(sbbs_idcs)
      sbbs_bead_names = np.array(sbbs_bead_names)

      proper_sbbs_indices = []
      if len(sbbs_idcs) > 3:
         sort_index = np.array([1, 0, 2, 3])
         for index in range(0, len(sbbs_idcs) - 3):
            if np.all(sbbs_bead_names[index:index + 4][sort_index] == np.array(['SC1', 'BB', 'BB', 'SC1'])):
               proper_sbbs_indices.append(sbbs_idcs[index:index + 4][sort_index])

      proper_sbbs_mul          = 2 * torch.ones(len(proper_sbbs_indices), dtype=torch.float32) # non-trainable
      self.proper_sbbs_phase   = torch.nn.Parameter(torch.zeros_like(proper_sbbs_mul))
      self.proper_sbbs_const   = torch.nn.Parameter(5. * torch.ones_like(proper_sbbs_mul))

      self.register_buffer('proper_sbbs_indices', torch.tensor(np.array(proper_sbbs_indices)))
      self.register_buffer('proper_sbbs_mul', proper_sbbs_mul)

      # - BBBB - #

      bbbb_idcs = []
      bbbb_bead_names = []
      for index, bead_name in enumerate(bead_names):
         if bead_name in ['BB']:
            bbbb_idcs.append(index)
            bbbb_bead_names.append(bead_name)
      bbbb_idcs = np.array(bbbb_idcs)

      proper_bbbb_indices = []
      if len(bbbb_idcs) > 3:
         for index in range(0, len(bbbb_idcs) - 3, 2):
            proper_bbbb_indices.append(bbbb_idcs[index:index + 4])

      proper_bbbb_mul          = 2 * torch.ones(len(proper_bbbb_indices), dtype=torch.float32) # non-trainable
      self.proper_bbbb_phase   = torch.nn.Parameter(torch.zeros_like(proper_bbbb_mul))
      self.proper_bbbb_const   = torch.nn.Parameter(5. * torch.ones_like(proper_bbbb_mul))

      self.register_buffer('proper_bbbb_indices', torch.tensor(np.array(proper_bbbb_indices)))
      self.register_buffer('proper_bbbb_mul', proper_bbbb_mul)

      # ----- NON-BONDED ----- #
      
      all_indices = np.triu_indices(num_beads, 1)
      first_bead  = bond_indices[:, 0]
      second_bead = bond_indices[:, 1]

      first_bead  = np.append(first_bead,  angle_indices[:, 0])
      second_bead = np.append(second_bead, angle_indices[:, 2])

      index = (
         (first_bead[:] * (first_bead[:] + 1)) / 2 + first_bead[:] * (num_beads - first_bead[:] - 1) + second_bead[:] - first_bead[:] - 1
      ).astype(int)
      self.nonbonded_indices = [[],[]]
      self.nonbonded_indices[0] = np.delete(all_indices[0],index,None)
      self.nonbonded_indices[1] = np.delete(all_indices[1],index,None)
      
      self.h_bond_selection = torch.tensor([15, 16], dtype=int)

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
      self.nonbonded_vals = torch.from_numpy(np.asanyarray([i for i in self.nonbonded_dict.values()]))

      self.nonbonded_type_keys = np.unique(np.sort(np.array([[bead_types[self.nonbonded_indices[0][i]], bead_types[self.nonbonded_indices[1][i]]] for i in range(len(self.nonbonded_indices[0]))]), axis=1), axis=0)

      self.nonbond_dict_index = []
      nonbonded_pairs = [[self.nonbonded_indices[0][i],self.nonbonded_indices[1][i]] for i in range(len(self.nonbonded_indices[0]))]
      self.nonbond_type_dict_index = []

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

      self.nonbond_dict_index = torch.asarray(self.nonbond_dict_index)

      self.nonbond_type_dict_index = torch.from_numpy(np.array(self.nonbond_type_dict_index))

      self.nonbonded_type_keys = torch.asarray(self.nonbonded_type_keys)

      self.nonbonded_indices = torch.from_numpy(np.array(self.nonbonded_indices))

      self.bead_radii = torch.nn.Parameter(torch.Tensor([1.2 for i in range(0,bead_types.max()+1)]) * pos2unit)

      bead_pair_selection = self.nonbonded_type_keys[self.nonbond_type_dict_index]
      self.register_buffer('bead_pair_selection', bead_pair_selection)

      self.bead_types = torch.nn.Parameter(torch.Tensor(bead_types))
      

      self.dispertion_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.30))
      # self.lj_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.500000e-01  ))

      # self.bond_H_lj_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 2.500000e-01 ))
      self.bond_H_strength_const = torch.nn.Parameter(torch.Tensor(self.nonbonded_vals * 1.00 )  * pos2unit)


      ## Charges needs to be per bead type not different for each bead ##
      self.e_0 = torch.nn.Parameter(torch.Tensor([8.8541878128e-3]))
      self.e_r = torch.nn.Parameter(torch.Tensor([1])) #########################[8.8541878128e-3]
      #########################################################################################################
      self.f_0 = torch.nn.Parameter(torch.Tensor([138.935458]))
      self.bead_charges_vals = torch.nn.Parameter(torch.reshape(torch.tensor(np.array(list(conf_bead_charges.values())), dtype=torch.float32), (-1,)))
      self.bead_charges_keys = np.asanyarray([i for i in conf_bead_charges.keys()])

      self.bead_charge_indices = []
      for bead_name in bead_idnames:
         self.bead_charge_indices.extend(np.where(self.bead_charges_keys == bead_name)[0])
      self.bead_charge_indices = torch.asarray(self.bead_charge_indices)
      self.bead_charge_indices = torch.asarray(self.bead_charge_indices)
      

   def bonds(self, bead_pos):
      bond_pos = bead_pos[:, self.bond_indices, :]
      bond_distances = bond_pos[..., 1, :] - bond_pos[..., 0, :]
      bond_norm = torch.norm(bond_distances, dim=-1)
      
      bond_energy = (
         0.5 * torch.abs(self.spring_constant_vals[self.bond_dist_index]) *
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
      )

      return dihedral_energy

   def proper_dih_sbbs(self, bead_pos):
      torsion = get_dihedrals(bead_pos, self.proper_sbbs_indices)

      proper_dih_energy = (
         torch.abs(self.proper_sbbs_const) *
         torch.max(
            torch.zeros_like(torsion),
            (1 + torch.cos(self.proper_sbbs_mul * torsion - self.proper_sbbs_phase)) - 0.05
         )
      )

      return proper_dih_energy
   
   def proper_dih_bbbb(self, bead_pos):
      torsion = get_dihedrals(bead_pos, self.proper_bbbb_indices)

      proper_dih_energy = (
         torch.abs(self.proper_bbbb_const) *
         torch.max(
            torch.zeros_like(torsion),
            (1 + torch.cos(self.proper_bbbb_mul * torsion - self.proper_bbbb_phase)) - 0.05
         )
      )

      return proper_dih_energy

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

         all_dist = bead_pos[..., None, :] - bead_pos[:, None, ...]
         nonbonded_dist = all_dist[:, self.nonbonded_indices[0], self.nonbonded_indices[1], :]
         nonbonded_norm = torch.norm(nonbonded_dist, dim=-1)
         # cutoff_indices = torch.where(nonbonded_norm > 1.2)
         # nonbonded_norm = nonbonded_norm * 
         ###########################
         x=nonbonded_norm
         p=12
         factor = 1.0 / self.r_max
         x = x * factor
      
         out = 1.0
         out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
         out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
         out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
         cutoff_matrix = out * (x < 1.0)
         #############################
         # cutoff_matrix = _poly_cutoff(nonbonded_norm, 1.2, p=12)
         # cutoff_matrix = torch.where(nonbonded_norm > 1.2, 0, 1.0)

         bead_pair_radius = self.bead_radii[self.bead_pair_selection].sum(dim=1)

         eps             = torch.abs(self.dispertion_const[self.nonbond_dict_index]) 
         sigma_over_r    = torch.div(bead_pair_radius, nonbonded_norm) * cutoff_matrix
         lj_energy       = 4 * torch.einsum('j,ij->ij', eps, torch.pow(sigma_over_r, 12) - torch.pow(sigma_over_r, 6))

         epsH            = torch.abs(self.bond_H_strength_const[self.nonbond_dict_index]) 
         sigma_over_rH   = torch.div(bead_pair_radius, nonbonded_norm)* cutoff_matrix
         Hbond_lj_energy = torch.einsum('j,ij->ij', epsH, torch.pow(sigma_over_rH, 6) - torch.pow(sigma_over_rH, 4))

         # h_bond_norm = torch.norm(all_dist[:, [10,10], [19,18], :], dim=-1)
         # epsH            = torch.abs(self.bond_H_strength_const[self.h_bond_selection]) 
         # sigma_over_rH   = torch.div(bead_pair_radius[self.h_bond_selection], h_bond_norm)
         # Hbond_lj_energy = torch.einsum('j,ij->ij', epsH, torch.pow(sigma_over_rH, 6) - torch.pow(sigma_over_rH, 4))

         charges = self.bead_charges_vals[self.bead_charge_indices]
         bead_charge = torch.einsum('i,j->ij', charges, charges)[self.nonbonded_indices[0], self.nonbonded_indices[1]] 
         coulumb_energy = torch.div(bead_charge, nonbonded_norm) * self.f_0 / self.e_r * cutoff_matrix #/ (4 * self.e_0 * torch.pi) #

         bond_energy = self.bonds(bead_pos)
         angle_energy = self.angles(bead_pos)
         dihedral_energy = self.dihedrals(bead_pos)
         proper_dih_sbbs_energy = self.proper_dih_sbbs(bead_pos)
         proper_dih_bbbb_energy = self.proper_dih_bbbb(bead_pos) 

         energies = (
            torch.sum(bond_energy) +
            torch.sum(angle_energy) +
            torch.sum(dihedral_energy) +
            torch.sum(proper_dih_sbbs_energy) +
            torch.sum(proper_dih_bbbb_energy)
            # torch.sum(coulumb_energy) +
            # torch.sum(lj_energy) +
            # torch.sum(Hbond_lj_energy)
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