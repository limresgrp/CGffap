import torch
import numpy as np
import os
import yaml

from typing import Dict
from pathlib import Path


class EquiValReporter(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.conf  = None
        self.config_angles = None
        self.sides = None
        self.config_bead_charge = None
        self.config_improper_dih = None

    def bondMapper(self, config_file_path):

        conf : Dict = yaml.safe_load(Path(config_file_path).read_text()) 

        Bond_for_bead_dict = {}

        bonds_mat = []
        frame = 0

        for root_bead, leaf_beads in conf.items():
            
            
            if root_bead in self.dataset["bead_idnames"]:
                bead_types = leaf_beads.split()
                Bond_for_bead_dict.update({root_bead : np.zeros(len(bead_types))})
                beads_to_bond = np.where(root_bead == self.dataset["bead_idnames"])

                for i in bead_types:
                    dist_mat = []

                    if i == "Skip":
                        continue

                    beads_to_consider = np.where(i == self.dataset["bead_idnames"])

                    # Get the bead that is closest to the one that have been chosen w.r.t. the positions matrix
                    
                    root_pos = self.dataset["bead_pos"][frame][beads_to_bond]
                    leaf_pos = self.dataset["bead_pos"][frame][beads_to_consider] 

                    for bead_pos in root_pos:
                        dist_mat.append(np.linalg.norm(leaf_pos - bead_pos, axis=1))

                    dist_mat = np.asarray(dist_mat)
                    
                    if i == root_bead :
                        # Getting diagonal indices
                        ind = np.diag_indices_from(dist_mat)

                        # Giving the diagonals inf value to avoid self assigned bonds in the computation
                        dist_mat[ind] = np.inf

                    # assign beads to be bonded with minimum distance
                    leafs_to_bond = dist_mat.argmin(axis=1)

                    # Generate the matrix of side chain bonds
                    for index in range(len(beads_to_bond[0])):
                        bonds_mat.append([beads_to_bond[0][index] , beads_to_consider[0][leafs_to_bond[index]] ])

        bonds_mat = np.asarray(bonds_mat)

        print("Side Chain Bond Matrix Shape" , bonds_mat.shape)

        BB_beads_to_bond = np.array((0))
        BB_bonds_mat = []
        initial = True
        for root_bead, leaf_beads in conf.items():
            
            if root_bead in self.dataset["bead_idnames"]:
                
                # Filter Backbone Beads
                if "_BB" in root_bead:    
                    
                    if initial == True:
                        initial = False
                        BB_beads_to_bond = np.asarray(np.where(root_bead == self.dataset["bead_idnames"]))
                    else:
                        temp = np.asarray(np.where(root_bead == self.dataset["bead_idnames"]))
                        BB_beads_to_bond = np.concatenate((BB_beads_to_bond, temp), axis = 1)  

        BB_beads_to_bond = sorted(BB_beads_to_bond[0])

        for i in range(len(BB_beads_to_bond)-1):
            BB_bonds_mat.append([BB_beads_to_bond[i],BB_beads_to_bond[i+1]])
            BB_bonds_mat.append([BB_beads_to_bond[i+1],BB_beads_to_bond[i]])


        # Combine SC and BB bonds in one Matrix
        all_bonds_mat = np.append(bonds_mat, BB_bonds_mat, axis=0)
        print("All bonds Mat Shape",all_bonds_mat.shape)

        # Sort the Bonds 
        sorted_all_bonds_mat = all_bonds_mat[all_bonds_mat[:, 0].argsort()]

        pos_bond_mat = self.dataset["bead_pos"][:,sorted_all_bonds_mat,:]
        # print(pos_bond_mat.shape)
        # print((pos_bond_mat[:,:,0,:] - pos_bond_mat[:,:,1,:]).shape)
        dist_bond_mat = np.linalg.norm(pos_bond_mat[:,:,0,:] - pos_bond_mat[:,:,1,:], axis=2)
        # print(dist_bond_mat.shape)
        dist_mean_mat = np.average(dist_bond_mat, axis=0).reshape(-1,1)
        # print(dist_mean_mat.shape)
        dist_mean_mat = np.append(sorted_all_bonds_mat.reshape(-1,2), dist_mean_mat, axis=1)

        dist_mean_mat = np.append(self.dataset["bead_idnames"][sorted_all_bonds_mat].reshape(-1,2), dist_mean_mat, axis=1)


        bead_bond_dict = {}
        for root_bead, leaf_bead, index_root, index_leaf, bond_dist in dist_mean_mat:
            # print(root_bead, leaf_bead)
            if root_bead in bead_bond_dict:
                flag = False

                for i in range(len(bead_bond_dict[root_bead])):
                    if leaf_bead in str(bead_bond_dict[root_bead][i]):
                        bead_bond_dict[root_bead][i].append(bond_dist)
                        # print(root_bead, leaf_bead)
                        flag = True

                if flag == False:
                    bead_bond_dict[root_bead].append([leaf_bead,bond_dist])
                    flag = False

            elif root_bead not in bead_bond_dict :
                bead_bond_dict.update({root_bead : []})
                bead_bond_dict[root_bead].append([leaf_bead,bond_dist])


        bead_bond_dict_mean = {}
        for root_bead, leaf_bead, index_root, index_leaf, bond_dist in dist_mean_mat:
            bead_bond_dict_mean[root_bead] =  {}
            for i in range(len(bead_bond_dict[root_bead])):
                # print(np.asarray(bead_bond_dict["SER_BB"][i][0]))
                temp = list(map(float, bead_bond_dict[root_bead][i][1:]))
                temp = np.mean(temp)
                key = bead_bond_dict[root_bead][i][0]
                bead_bond_dict_mean[root_bead].update({key : temp})


        num_issues = 0
        to_remove = []
        to_replace = []
        for root_bead in bead_bond_dict_mean :
            for leaf_bead in bead_bond_dict_mean[root_bead]:
                # print(root_bead,leaf_bead)
                if leaf_bead not in bead_bond_dict_mean or root_bead not in bead_bond_dict_mean[leaf_bead] :
                    print("Between",root_bead,"and",leaf_bead,"no bond found")
                    to_remove.append([root_bead,leaf_bead])
                    num_issues += 1
                    continue

                elif np.abs(bead_bond_dict_mean[root_bead][leaf_bead] - bead_bond_dict_mean[leaf_bead][root_bead]) > 0.0001:
                    print("Distance mismatch found!")
                    print(root_bead,"to",leaf_bead,"mean distance = ",bead_bond_dict_mean[root_bead][leaf_bead],"vs",leaf_bead,"to",root_bead,"mean distance = ",bead_bond_dict_mean[leaf_bead][root_bead])
                    num_issues += 1
                    to_replace.append([root_bead,leaf_bead])
                    print("Difference mean distance =",np.abs(bead_bond_dict_mean[root_bead][leaf_bead] - bead_bond_dict_mean[leaf_bead][root_bead]))

        print("Bonds to be removed",to_remove,"Bonds to be replaced with correct value",to_replace)
        print("Number of issued found:",num_issues)

        # for bead in to_remove:
        #     try:
        #         del bead_bond_dict_mean[bead[0]][bead[1]]
        #         num_issues -= 1
        #     except:
        #         print(bead,"removed")
        #     try:
        #         del bead_bond_dict_mean[bead[1]][bead[0]]
        #         num_issues -= 1
        #     except:
        #         print(bead,"removed")

        for bead in to_replace:
            if(bead_bond_dict_mean[bead[0]][bead[1]] > bead_bond_dict_mean[bead[1]][bead[0]]):
                bead_bond_dict_mean[bead[0]][bead[1]] = bead_bond_dict_mean[bead[1]][bead[0]]
                num_issues -= 1
                print(bead,"replaced")
            else:
                bead_bond_dict_mean[bead[1]][bead[0]] = bead_bond_dict_mean[bead[0]][bead[1]]
                num_issues -= 1
                

            
        print("Number of issues remain:",num_issues)

        config_bonds = []
        conf_dict = {}
        for root_bead in bead_bond_dict_mean:
            for leaf_bead in bead_bond_dict_mean[root_bead]:
                if (str(root_bead + '-' + leaf_bead) and str(leaf_bead + '-' + root_bead)) not in conf_dict.keys():
                    conf_dict[str(root_bead + '-' + leaf_bead)] = bead_bond_dict_mean[root_bead][leaf_bead]
                    config_bonds.append("{}-{} : {}".format(root_bead,leaf_bead, bead_bond_dict_mean[root_bead][leaf_bead]))
        
        self.conf = conf_dict
        self.dataset['bond_indices'] = sorted_all_bonds_mat

        return True



    def angleMapper(self, conf_angles_path):

        bonds = self.dataset["bond_indices"]
        angle_beads = []
        angles = {}
        frame = 0
        for index in range(bonds.max()+1):
            i, j= np.where(bonds == index)
            angle_beads.append([index,bonds[i,np.abs((j-1)%2)]])
            # angle_beads.append([index, np.delete(bonds[i][:,1], np.where(bonds[i][:,1] == index))])
            # print(angle_beads[index])
            for k in range(len(angle_beads[index][1])):
                pair  = (k+1)%len(angle_beads[index][1])
                bond = [angle_beads[index][1][k],angle_beads[index][0],angle_beads[index][1][pair]]
                if bond[0] == bond[2]:
                    continue
                else:
                    # print(bond)
                    #insert 3d angle calculation
                    vector1 = self.dataset["bead_pos"][:,angle_beads[index][1][k]] - self.dataset["bead_pos"][:,angle_beads[index][0]]
                    vector2 = self.dataset["bead_pos"][:,angle_beads[index][1][pair]] - self.dataset["bead_pos"][:,angle_beads[index][0]]
                    # print(vector1)
                    # print(np.linalg.norm(vector1, axis=1))
                    unit_vec1 = vector1[:] / np.linalg.norm(vector1, axis=-1)[:,None]
                    unit_vec2 = vector2[:] / np.linalg.norm(vector2, axis=-1)[:,None]
                    # print(unit_vec2.shape , unit_vec1.shape)
                    # angle = np.arccos(np.sum(unit_vec1 * unit_vec2, axis=-1))
                    angle = np.average(np.arccos(np.sum(unit_vec1 * unit_vec2, axis=-1)))
                    # print(angle)
                    # print(vector1[1]/(np.linalg.norm(vector1[1])))
                    # # print(vector1, vector2, angle)


                    if bond[0] < bond[2]:
                        angles[str(bond[0]) + "-" + str(bond[1]) + "-" + str(bond[2])] = angle
                    else:
                        angles[str(bond[2]) + "-" + str(bond[1]) + "-" + str(bond[0])] = angle

                    # print(dataset['bead_names'][bond[0]], dataset['bead_names'][bond[1]], dataset['bead_names'][bond[2]])

                    # if bond[0] < bond[2]:
                    #     angles.update({dataset['bead_names'][bond[0]] : {dataset['bead_names'][bond[1]] : {dataset['bead_names'][bond[2]] : [angle]}}})
                    # else:
                    #     angles.update({dataset['bead_names'][bond[2]] : {dataset['bead_names'][bond[1]] : {dataset['bead_names'][bond[0]] : [angle]}}})

        angle_indices = np.array([list(map(int, item.split('-'))) for item in angles.keys()])


        angle_dic : list = yaml.safe_load(Path(conf_angles_path).read_text())
        angle_array = np.array([values for values in angles.values()])
        for i in self.dataset['bead_idnames'][angle_indices]:
            rows = np.where(self.dataset['bead_idnames'][angle_indices[:,1]].__eq__(i[1]))[0]

            sides = []
            sides = np.union1d(np.intersect1d(np.where(self.dataset['bead_idnames'][angle_indices[rows,0]].__eq__(i[2]))[0],np.where(self.dataset['bead_idnames'][angle_indices[rows,2]].__eq__(i[0]))[0]), np.intersect1d(np.where(self.dataset['bead_idnames'][angle_indices[rows,0]].__eq__(i[0]))[0], np.where(self.dataset['bead_idnames'][angle_indices[rows,2]].__eq__(i[2]))[0]))
            self.sides = sides
            key = str(angle_indices[rows[sides]][0,0]) + "-" +  str(angle_indices[rows[sides]][0,1]) + "-" + str(angle_indices[rows[sides]][0,2])

            if str(i[2] + "-" + i[1] + "-" + i[0]) not in angle_dic:
                angle_dic[i[0] + "-" + i[1] + "-" + i[2]] = np.average(np.asanyarray(angles[key]))

        config_angles_dict = list(zip(angle_dic.keys(), angle_dic.values()))
        config_angles = []
        for i,j in config_angles_dict:
            config_angles.append("{} : {}".format(i,j))
        
        self.config_angles = config_angles

        return True
    
    def improperDihedralMapper(self):
        dihedral_conf = ['TRP_BB', 6, 'TYR_BB', 5]
        indices = []
        bead_dih = []
        for bead in dihedral_conf:
            if isinstance(bead, int):
                for index in indices[-1]:
                    temp = []
                    for i in range(bead):
                        temp.append(index + i)
                    bead_dih.append(temp)
                    
                
                print(bead_dih)
            else:
                indices.append(np.where(bead == self.dataset['bead_idnames'])[0])
                print(indices[-1])

        dihedrals = {}
        for residue in bead_dih:
            print(residue)
            if len(residue) == 6:
                residue.pop(0)
                residue.pop(2)
                print(residue)
                poses  = self.dataset['bead_pos'][:,residue]
                # dihedral.append()
                
            elif len(residue) == 5:
                residue.pop(0)
                residue = np.asanyarray(residue)
                print(residue)
                poses  = self.dataset['bead_pos'][:,residue]

            
            # print(poses[0])
            plane1 = [poses[:,0], poses[:,1], poses[:,2]]
            plane2 = [poses[:,1], poses[:,2], poses[:,3]]
            # dihedral.append()
            # print(plane1[1], plane2[0])
            vector1 = plane1[0] - plane1[1]
            vector2 = plane1[2] - plane1[1]
            vector3 = plane2[0] - plane2[1]
            vector4 = plane2[2] - plane2[1]

            unit_vec1 = vector1[:] / np.linalg.norm(vector1, axis=-1)[:,None]
            unit_vec2 = vector2[:] / np.linalg.norm(vector2, axis=-1)[:,None]
            unit_vec3 = vector3[:] / np.linalg.norm(vector3, axis=-1)[:,None]
            unit_vec4 = vector4[:] / np.linalg.norm(vector4, axis=-1)[:,None]

            binorm1 = np.cross(unit_vec1, unit_vec2, axis=-1) 
            binorm2 = np.cross(unit_vec3, unit_vec4, axis=-1) 
            binorm1 /=  np.linalg.norm(binorm1, axis=-1)[:,None]
            binorm2 /=  np.linalg.norm(binorm2, axis=-1)[:,None]

            # np.dot(binorm1[:,None], binorm2[:,None])
            # print(unit_vec1[0], unit_vec2[0])
            # print("binorm = ",binorm1.shape, binorm1[0])

            torsion = np.average(np.arccos(np.sum(binorm1[0] * binorm2[0], axis=-1)) - np.pi)
            print(vector1[0], vector2[0], vector3[0],vector4[0])
            dihedrals[str(residue[0]) + "-" + str(residue[1]) + "-" + str(residue[2]) + "-" + str(residue[3])] = torsion

        dih_indices = np.asanyarray([list(map(int, item.split('-'))) for item in dihedrals.keys()])
        self.dataset['bead_names'][dih_indices[:,0]]

        conf_dihedrals: list = yaml.safe_load(Path("config files/config.naive.per.bead.type.dihedrals.yaml").read_text())

        dih_dic = conf_dihedrals
        dih_array = np.asanyarray([values for values in dihedrals.values()])
        print(dih_array)
        for i in self.dataset['bead_idnames'][dih_indices]:
            rows = np.where(self.dataset['bead_idnames'][dih_indices[:,0]].__eq__(i[0]))[0]
            print(i,rows,dih_indices[rows,0], self.sides)
            
            dih_dic[i[0] + "-" + i[1] + "-" + i[2] + "-" + i[3]] = np.mean(np.asanyarray(dih_array[rows]))

        config_dihedrals_per_bead = []
        for root_bead in dih_dic:
            config_dihedrals_per_bead.append("{} : {}".format(root_bead, dih_dic[root_bead]))
        config_dihedrals_per_bead

        self.config_improper_dih = config_dihedrals_per_bead

        return True
    
    def beadChargeMapper(self):
        bead_charges = []
        for bead in self.dataset['bead_idnames']:
            if bead == 'GLU_SC1' or bead == 'ASP_SC1' :
                bead_charges.append(-1)
            elif bead == 'ARG_SC2' or bead == 'HIS_SC3' or bead == 'LYS_SC2':
                bead_charges.append(+1)
            else:
                bead_charges.append(0)
        print(bead_charges)
        self.dataset['bead_charges'] = np.asanyarray(bead_charges).reshape(-1,1)

        beadcharges = list(zip(self.dataset['bead_idnames'], self.dataset['bead_charges']))
        bead_charge_dict = {}
        for i, j in beadcharges:
            bead_charge_dict[i] = j

        self.dataset['bead_charge_dict'] = bead_charge_dict

        config_BeadCharges_per_bead = []
        for root_bead in bead_charge_dict:
            config_BeadCharges_per_bead.append("{} : {}".format(root_bead, bead_charge_dict[root_bead][0]))
        self.config_bead_charge = config_BeadCharges_per_bead

        return True

    def getDataset(self):

        return self.dataset

    def reportEquiVals(self, reportPath = ''):

        output_file = os.path.join(reportPath, "config.dihedrals.yaml")

        with open(output_file, 'w') as f:
            for line in self.config_improper_dih:
                f.write(f"{line}\n")
        print(f"{output_file} successfully saved!")


    def getBonds(self):
        return self.conf


                                        