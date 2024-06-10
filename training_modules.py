from typing import Optional
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
         num_bead_types: Optional[int] = None,
      ):
      super().__init__()

      # ----- MISC ----- #
      
      self.r_max = 10. * pos2unit

      # ----- DATASET ----- #

      bead_idnames = dataset['bead_idnames']
      bead_names   = dataset['bead_names']
      self.register_buffer('bead_types', torch.from_numpy(dataset['bead_types']))
      num_bead_types = num_bead_types or self.bead_types.max() + 1

      bond_indices:         np.ndarray = dataset['bond_indices']
      angle_indices:        np.ndarray = dataset['angle_indices']
      improper_dih_indices: np.ndarray = dataset['improper_dih_indices']

      # ----- BONDED ----- #

      # --- BONDS --- #

      bond_indices = torch.from_numpy(bond_indices)
      self.register_buffer("bond_indices", bond_indices)
      
      self.equ_val_bond_dist_keys = np.array(list(conf_bonds.keys()))
      self.register_buffer("bond_equ_vals", torch.tensor(list(conf_bonds.values())) * pos2unit)
      self.bond_spring_constant_vals   = torch.nn.Parameter(20. * torch.ones_like(self.bond_equ_vals) * eng2unit)

      bond_dist_index = []
      for bond_ids in bond_indices:
         if str(bead_idnames[bond_ids[0]] + '-' + bead_idnames[bond_ids[1]]) in self.equ_val_bond_dist_keys:
            key = str(bead_idnames[bond_ids[0]] + '-' + bead_idnames[bond_ids[1]]) 
         elif str(bead_idnames[bond_ids[1]] + '-' + bead_idnames[bond_ids[0]]) in self.equ_val_bond_dist_keys:
            key = str(bead_idnames[bond_ids[1]] + '-' + bead_idnames[bond_ids[0]]) 

         bond_dist_index.append(np.where(self.equ_val_bond_dist_keys == key)[0][0])
      
      self.bond_dist_index = torch.tensor(bond_dist_index)

      # --- ANGLES --- #
      
      angle_indices = torch.from_numpy(angle_indices)
      self.register_buffer("angle_indices", angle_indices)

      self.equ_val_angles_keys        = np.array(list(conf_angles.keys()))
      self.register_buffer('angle_equ_vals', torch.tensor(list(conf_angles.values())))
      self.angle_spring_constant_vals = torch.nn.Parameter(2. * torch.ones_like(self.angle_equ_vals) * eng2unit)

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
      self.improper_dihedral_constant_vals    = torch.nn.Parameter(.2 * torch.ones_like(self.equ_val_dihedrals_vals) * eng2unit)

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
      proper_sbbs_indices = np.stack(proper_sbbs_indices, axis=0)

      proper_sbbs_mul          = 2 * torch.ones(len(proper_sbbs_indices), dtype=torch.float32) # non-trainable
      self.proper_sbbs_phase   = torch.nn.Parameter(torch.zeros_like(proper_sbbs_mul))
      self.proper_sbbs_const   = torch.nn.Parameter(5. * torch.ones_like(proper_sbbs_mul))

      self.register_buffer('proper_sbbs_indices', torch.tensor(proper_sbbs_indices))
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
      proper_bbbb_indices = np.stack(proper_bbbb_indices, axis=0)

      proper_bbbb_mul          = 2 * torch.ones(len(proper_bbbb_indices), dtype=torch.float32) # non-trainable
      self.proper_bbbb_phase   = torch.nn.Parameter(torch.zeros_like(proper_bbbb_mul))
      self.proper_bbbb_const   = torch.nn.Parameter(5. * torch.ones_like(proper_bbbb_mul))

      self.register_buffer('proper_bbbb_indices', torch.from_numpy(proper_bbbb_indices))
      self.register_buffer('proper_bbbb_mul', proper_bbbb_mul)

      # ----- NON-BONDED ----- #

      # --- LENNARD-JONES --- #

      # - We use type 2 combination rule of Gromacs - #
      # -     sigma12 = 0.5 * (sigma1 + sigma2)     - #
      # -         eps12 = sqrt(eps1 * eps2)         - #

      self.bead_radii         = torch.nn.Parameter(1.2 * torch.ones(num_bead_types, dtype=torch.get_default_dtype()) * pos2unit)
      self.dispertion_const   = torch.nn.Parameter(0.5 * torch.ones(num_bead_types, dtype=torch.get_default_dtype()) * eng2unit)

      self.H_bead_radii       = torch.nn.Parameter(2.5 * torch.ones(num_bead_types, dtype=torch.get_default_dtype()) * pos2unit)
      self.H_dispertion_const = torch.nn.Parameter(0.2 * torch.ones(num_bead_types, dtype=torch.get_default_dtype()) * eng2unit)

      # --- COULOMB --- #
      
      # Join f and eps_r in a single parameter called f_r
      self.f_r = torch.nn.Parameter(torch.Tensor([3.3206])) # 3.3206 (A * kcal) / (mol * e**2) == 138.935 (nm * kJ) / (mol * e**2)
      bead_charges_vals = torch.zeros(num_bead_types, dtype=torch.get_default_dtype())

      for bead_idname, charge_value in conf_bead_charges.items():
         bead_charges_vals[self.bead_types[np.where(bead_idnames == bead_idname)[0][0]]] = charge_value.item()
      
      self.bead_charges_vals = torch.nn.Parameter(bead_charges_vals)
   
   @torch.jit.export
   def set_bead_types(self, bead_types: torch.Tensor):
      self.bead_types = bead_types

   def compute_nonbonded_params(self, bead_pos: torch.Tensor):

      num_beads = bead_pos.shape[1]

      # - Bonded pairs to exclude from nonbonded - #
      bonded_pair_idcs = torch.stack([torch.sort(pair).values for pair in torch.cat([
            self.bond_indices,
            self.angle_indices[:, ::2],
            self.proper_sbbs_indices[:, ::3],
            self.proper_bbbb_indices[:, ::3],
         ], dim=0)
      ], dim=0)

      nonbonded_pair_idcs = torch.triu_indices(num_beads, num_beads, offset=1, device=bonded_pair_idcs.device).T
      nonbonded_indices = remove_intersecting_rows(nonbonded_pair_idcs, bonded_pair_idcs)
      all_dist = bead_pos[..., None, :] - bead_pos[:, None, ...]
      nonbonded_dist = all_dist[:, nonbonded_indices[:, 0], nonbonded_indices[:, 1], :]
      nonbonded_norm = torch.norm(nonbonded_dist, dim=-1)
      cutoff_mask = torch.any(nonbonded_norm <= self.r_max, dim=0)
      self.nonbonded_indices = nonbonded_indices[cutoff_mask]
      
      nonbonded_param_index = self.bead_types[self.nonbonded_indices]
      self.bead_pair_radii = self.bead_radii[nonbonded_param_index].sum(dim=1)
      self.bead_pair_eps = self.dispertion_const[nonbonded_param_index].prod(dim=1).abs().sqrt()

      self.H_bead_pair_radii = self.H_bead_radii[nonbonded_param_index].sum(dim=1)
      self.H_bead_pair_eps = self.H_dispertion_const[nonbonded_param_index].prod(dim=1).abs().sqrt()

      self.bead_pair_charges = self.bead_charges_vals[nonbonded_param_index].prod(dim=-1)
      return nonbonded_norm[:, cutoff_mask]

   def bonds(self, bead_pos):
      bond_pos = bead_pos[:, self.bond_indices, :]
      bond_distances = bond_pos[..., 1, :] - bond_pos[..., 0, :]
      bond_norm = torch.norm(bond_distances, dim=-1)
      
      bond_energy = (
         0.5 * torch.abs(self.bond_spring_constant_vals[self.bond_dist_index]) *
         torch.max(
            torch.zeros_like(bond_norm),
            torch.pow(bond_norm - self.bond_equ_vals[self.bond_dist_index], 2) - 0.005
         )
      )
      return bond_energy
      
   def angles(self, bead_pos):
      angles = get_angles(bead_pos, self.angle_indices)
      
      angle_energy = (
         0.5 * torch.abs(self.angle_spring_constant_vals[self.angle_rad_index]) *
         torch.max(
            torch.zeros_like(angles),
            torch.pow(angles - self.angle_equ_vals[self.angle_rad_index], 2) - 0.05
         )
      )

      return angle_energy
   
   def improper_dih(self, bead_pos):
      torsion = get_dihedrals(bead_pos, self.improper_dih_indices) 

      dihedral_energy = (
         0.5 * torch.abs(self.improper_dihedral_constant_vals[self.dih_rad_index]) *
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

         elif i in ['bond_equ_vals', 'bond_spring_constant_vals']:
           force_field[i]= dict(zip(self.equ_val_bond_dist_keys, self.state_dict()[i]))

         elif i in ['angle_equ_vals', 'angle_spring_constant_vals']:
           force_field[i]= dict(zip(self.equ_val_angles_keys, self.state_dict()[i]))

         elif i in ['improper_dihedral_constant_vals']:
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
            positions[i,k] is the position (default unit is angstrom) of spatial dimension k of particle i

         Returns
         -------
         potential : torch.Scalar
            The potential energy in energy units (default is in kcal/mol)
         """
         
         # convert as if it has batches
         if len(positions.shape) == 2:
            positions = positions[None, ...]

         bead_pos = positions

         bond_energy = self.bonds(bead_pos)
         angle_energy = self.angles(bead_pos)
         improper_dih_energy = self.improper_dih(bead_pos)
         proper_dih_sbbs_energy = self.proper_dih_sbbs(bead_pos)
         proper_dih_bbbb_energy = self.proper_dih_bbbb(bead_pos)

         nonbonded_norm = self.compute_nonbonded_params(bead_pos)
         
         sigma_over_r   = torch.div(self.bead_pair_radii, nonbonded_norm)
         lj_energy      = 4 * torch.einsum('j,ij->ij', self.bead_pair_eps, torch.pow(sigma_over_r, 12) - torch.pow(sigma_over_r, 6))

         H_sigma_over_r = torch.div(self.H_bead_pair_radii, nonbonded_norm)
         H_lj_energy    = 4 * torch.einsum('j,ij->ij', self.H_bead_pair_eps, torch.pow(H_sigma_over_r, 6) - torch.pow(H_sigma_over_r, 4))

         coulumb_energy = self.f_r * torch.div(self.bead_pair_charges, nonbonded_norm)

         energies = (
            torch.sum(bond_energy) +
            torch.sum(angle_energy) +
            torch.sum(improper_dih_energy) +
            torch.sum(proper_dih_sbbs_energy) +
            torch.sum(proper_dih_bbbb_energy) +
            torch.sum(lj_energy) +
            torch.sum(H_lj_energy) +
            torch.sum(coulumb_energy)
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

@torch.jit.script
def remove_intersecting_rows(arrA: torch.Tensor, arrB: torch.Tensor):
   m = (arrA[:, None] == arrB).all(-1).any(1)
   return arrA[~m]