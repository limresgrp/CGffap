from torch.utils.data import Dataset , DataLoader
import torch
from torch.optim import Adam, LBFGS
from torch.nn import MSELoss, L1Loss
import torch.nn as nn

from sklearn.model_selection import train_test_split

import yaml
from pathlib import Path
from typing import List, Union, Optional, Dict

import numpy as np
import os

import training_modules as tm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class CGDataset(Dataset):
    def __init__(self, npzdataset, stride = 1):
        super().__init__()
        self.data = {}
        for k, v in dict(npzdataset).items():
            if k in ['bead_pos', 'bead_forces']: #,'bead_types', 'bead_mass'
                self.data[k] = torch.from_numpy(v[::stride]).to('cuda') # data truncated to 10000

    def __len__(self):
        return len(self.data['bead_pos'])
    
    def __getitem__(self, index):
        datum = {}
        for k, v in self.data.items():
            if k in ['bead_pos', 'bead_forces']:
                datum[k] = v[index]
            else:
                datum[k] = v
        return datum
    
class TrainSystem(torch.nn.Module):
    def __init__(self, dataset, conf_bonds, conf_angles, conf_dihedrals, conf_bead_charges):
        super().__init__()

        self.potential = tm.ForceModel(dataset, conf_bonds, conf_angles, conf_dihedrals, conf_bead_charges)

        self.model  = tm.ForceMapper(self.potential)

        self.train_pos, self.valid_pos, self.train_force , self.valid_force = train_test_split(dataset['bead_pos'], dataset['bead_forces'], test_size=0.10, random_state=42)

        self.loss_matrix = []
        self.per_frame = []  
        self.train = None
        self.valid = None
        self.zero = None
        self.losswith0 = []
        self.batched_forces_plot = []
        self.batched_val_forces_plot = []
        self.initialguess = []

    def initiateTraining(self, train_steps = 10, batch_size = 10, num_workers = 0, dataset: dict = None, patience = 5, model_name = 'best_model'):
        self.model.train()
        writer = SummaryWriter()

        base_params = [p for name, p in self.named_parameters() if name not in [ 'potential.angle_spring_constant_vals', 
                                                                            'potential.dihedral_const_vals', 
                                                                            'potential.dispertion_const',
                                                                            'potential.bead_radii',
                                                                            'potential.bead_charges_vals',
                                                                            'potential.e_r',
                                                                            'potential.f_0']]
        
        charge_params = [p for name, p in self.named_parameters() if name in ['potential.bead_charges_vals',
                                                                              'potential.e_r',
                                                                              'potential.f_0']]

        constrained_params = [p for name, p in self.named_parameters() if name in [ 'potential.angle_spring_constant_vals', 
                                                                            'potential.dihedral_const_vals', 
                                                                            'potential.dispertion_const',
                                                                            'potential.bead_radii',
                                                                            ]]

        optimizer =  Adam([
                {'params': constrained_params, 'lr': 5e-3, 'weight_decay': 0},
                {'params': charge_params, 'lr': 1e-5, 'weight_decay': 0},
                {'params': base_params}
            ], lr=5e-2) #1e-3, , weight_decay = 1e-5
        loss_fn = MSELoss() # L1Loss() #    LBFGS(self.model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None) #
        dataset_train = dataset
        dataset_train['bead_pos'] = self.train_pos
        dataset_train['bead_forces'] = self.train_force

        data = CGDataset(dataset_train, stride=1)
        loader = DataLoader(
            data,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle=True
        )

        dataset_val = dataset
        dataset_val['bead_pos'] = self.valid_pos
        dataset_val['bead_forces'] = self.valid_force

        val_loader = DataLoader(CGDataset(dataset_val),batch_size = len(self.valid_force), num_workers = num_workers )


        train_losses = []
        val_losses = []
        zero_losses = []
        best_val_loss = float('inf')  # Initialize with a large value
        patience = patience  # Number of ep
        patience_counter = 0
        reps = 10

        # mean_zeroloss = loss_fn(batch['bead_forces']*0, batch['bead_forces'] ).detach().cpu().numpy()
        # mean_zeroloss_val = loss_fn(val_batch['bead_forces']*0, val_batch['bead_forces'] ).detach().cpu().numpy()

        mean_zeroloss = loss_fn(torch.Tensor(dataset_train['bead_forces'])*0, torch.Tensor(dataset_train['bead_forces']) ).detach().cpu().numpy()
        mean_zeroloss_val = loss_fn(torch.Tensor(dataset_val['bead_forces'])*0, torch.Tensor(dataset_val['bead_forces']) ).detach().cpu().numpy()


        for m_epoch in range(1,train_steps):
            if m_epoch %25 == 0:
                torch.jit.script(self.potential).save('Models/' + model_name + f'{m_epoch}.pt')

            loss_plot = []


             # Validation
            self.model.eval()
            val_loss_plot = []
            for val_batch in val_loader:
                val_out = self.model(val_batch['bead_pos'])
                val_loss = loss_fn(val_out , val_batch['bead_forces'] )
                self.batched_val_forces_plot = [val_out.detach().cpu().numpy() , val_batch['bead_forces'].detach().cpu().numpy()]
                val_loss_plot.append(val_loss.detach().cpu().numpy())


            # Training
            self.model.train()
            for batch in loader:
                # for epoch in range(0,10):
                # for batch in loader:

                optimizer.zero_grad()
                out = self.model(batch['bead_pos']) 
                loss = loss_fn(out, batch['bead_forces'])
                # writer.add_scalar("Loss/train", loss, m_epoch)
                if m_epoch == 1:
                    self.initialguess.append([out.detach().cpu().numpy()  , batch['bead_forces'].detach().cpu().numpy()])
                loss_plot.append(loss.detach().cpu().numpy())
                loss.backward()

                # def closure():
                #     optimizer.zero_grad()
                #     output = self.model(batch['bead_pos'])
                #     loss = loss_fn(output, batch['bead_forces'])
                #     loss.backward()
                #     return loss

                optimizer.step()
                for n, p in self.model.named_parameters():
                    # if n in [ 'module.angle_spring_constant_vals']: #, 'module.dihedral_const_vals', 'module.proper_dih_const', 'module.proper_dih_const_BB']: #, 'module.lj_const','module.bond_H_strength_const', 'module.dispertion_const'
                    #     p.data = p.data.clamp_(min=500)
                    if n in ['module.bead_charges_vals']:
                        p.data = p.data.clamp_(min=-1, max=1)
                    if n in ['module.equ_val_angles_vals', 'module.proper_phase_shift', 'module.proper_phase_shift_BB']:
                        p.data = p.data.clamp_(min=0, max = torch.pi)
                    if n in ['module.bead_radii']:
                        p.data = p.data.clamp_(min=0.12)
                    # self.per_frame.append(loss_plot)
                # self.losswith0.append(losswith0single)
            # loss_matrix.append(np.mean(loss_plot))
            # print("loss = ",np.mean(loss_plot), "epoch =", m_epoch)

            
            

            # Print and save results
            mean_train_loss = np.mean(loss_plot)
            mean_val_loss = np.mean(val_loss_plot)
            # loss_matrix.append(mean_train_loss)
            print(f"Epoch {m_epoch}: Train Loss = {mean_train_loss:.3f}, Val Loss = {mean_val_loss:.3f}, Zero Loss Train = {mean_zeroloss:.3f}, Zero Loss Val = {mean_zeroloss_val:.3f}")

                # Store losses for plotting
            train_losses.append(mean_train_loss)
            val_losses.append(mean_val_loss)

            # Save model checkpoint if validation loss improves
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.jit.script(self.potential).save(model_name+'.pt')
                self.batched_forces_plot = [out.detach().cpu().numpy()  , batch['bead_forces'].detach().cpu().numpy()]
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            # Check for early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {m_epoch} epochs.")
                break

        writer.flush()
        self.train =np.array(train_losses).reshape(-1)
        self.valid =np.array(val_losses).reshape(-1)
            
        return self.model
    
    def plotLosses(self, truncate = 0):
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, len(self.train[truncate:])), self.train[truncate:], label='Training Loss')
        plt.plot(range(0, len(self.valid[truncate:])), self.valid[truncate:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (kj mol^-1 nm^-1)^2')
        plt.title('Training and Validation Loss Over Epochs')
        # Place legend outside the box
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # plt.savefig("lossvsepochChignolinfoldednandunfolded177epochsearlystopped", format = "pdf")
        plt.show()

    def plotForceMathing(self, bead_index = 0, to_frame = 100):

        plt.figure(figsize=(10, 5))

        # x-axis plot
        fig1, ax1 = plt.subplots()
        ax1.plot(range(0, to_frame), self.batched_forces_plot[1][:to_frame,bead_index,0], label='Target Force x')
        ax1.plot(range(0, to_frame), self.batched_forces_plot[0][:to_frame,bead_index,0], label='Trained Force x')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Force')
        ax1.set_title('X Force Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # y-axis plot
        fig2, ax2 = plt.subplots()
        ax2.plot(range(0, to_frame), self.batched_forces_plot[1][:to_frame,bead_index,1], label='Target Force y')
        ax2.plot(range(0, to_frame), self.batched_forces_plot[0][:to_frame,bead_index,1], label='Trained Force y')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Force')
        ax2.set_title('Y Force Comparison')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # z-axis plot
        fig1, ax3 = plt.subplots()
        ax3.plot(range(0, to_frame), self.batched_forces_plot[1][:to_frame,bead_index,2], label='Target Force z')
        ax3.plot(range(0, to_frame), self.batched_forces_plot[0][:to_frame,bead_index,2], label='Trained Force z')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Force')
        ax3.set_title('Z Force Comparison')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()

    def plotValForceMathing(self, bead_index = 0, to_frame =100):

        plt.figure(figsize=(10, 5))

        # x-axis plot
        fig1, ax1 = plt.subplots()
        ax1.plot(range(0, to_frame), self.batched_val_forces_plot[1][:to_frame,bead_index,0], label='Target Force x')
        ax1.plot(range(0, to_frame), self.batched_val_forces_plot[0][:to_frame,bead_index,0], label='Trained Force x')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Force')
        ax1.set_title('X Force Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # y-axis plot
        fig2, ax2 = plt.subplots()
        ax2.plot(range(0, to_frame), self.batched_val_forces_plot[1][:to_frame,bead_index,1], label='Target Force y')
        ax2.plot(range(0, to_frame), self.batched_val_forces_plot[0][:to_frame,bead_index,1], label='Trained Force y')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Force')
        ax2.set_title('Y Force Comparison')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # z-axis plot
        fig1, ax3 = plt.subplots()
        ax3.plot(range(0, to_frame), self.batched_val_forces_plot[1][:to_frame,bead_index,2], label='Target Force z')
        ax3.plot(range(0, to_frame), self.batched_val_forces_plot[0][:to_frame,bead_index,2], label='Trained Force z')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Force')
        ax3.set_title('Z Force Comparison')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()

    def plotInitialForceGuess(self, bead_index = 0, to_frame = 100):

        plt.figure(figsize=(10, 5))

        # x-axis plot
        fig1, ax1 = plt.subplots()
        ax1.plot(range(0, to_frame), self.initialguess[0][1][:to_frame,bead_index,0], label='Target Force x')
        ax1.plot(range(0, to_frame), self.initialguess[0][0][:to_frame,bead_index,0], label='Trained Force x')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Force')
        ax1.set_title('X Force Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # y-axis plot
        fig2, ax2 = plt.subplots()
        ax2.plot(range(0, to_frame), self.initialguess[0][1][:to_frame,bead_index,1], label='Target Force y')
        ax2.plot(range(0, to_frame), self.initialguess[0][0][:to_frame,bead_index,1], label='Trained Force y')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Force')
        ax2.set_title('Y Force Comparison')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # z-axis plot
        fig1, ax3 = plt.subplots()
        ax3.plot(range(0, to_frame), self.initialguess[0][1][:to_frame,bead_index,2], label='Target Force z')
        ax3.plot(range(0, to_frame), self.initialguess[0][0][:to_frame,bead_index,2], label='Trained Force z')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Force')
        ax3.set_title('Z Force Comparison')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()


    def plotForceMathingByFrame(self, frame = 0):

        plt.figure(figsize=(10, 5))

        # x-axis plot
        fig1, ax1 = plt.subplots()
        ax1.plot(range(0, self.batched_forces_plot[0].shape[1]), self.batched_forces_plot[1][frame,:,0], label='Target Force x')
        ax1.plot(range(0, self.batched_forces_plot[0].shape[1]), self.batched_forces_plot[0][frame,:,0], label='Trained Force x')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Force')
        ax1.set_title('X Force Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # y-axis plot
        fig2, ax2 = plt.subplots()
        ax2.plot(range(0, self.batched_forces_plot[0].shape[1]), self.batched_forces_plot[1][frame,:,1], label='Target Force y')
        ax2.plot(range(0, self.batched_forces_plot[0].shape[1]), self.batched_forces_plot[0][frame,:,1], label='Trained Force y')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Force')
        ax2.set_title('Y Force Comparison')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # z-axis plot
        fig1, ax3 = plt.subplots()
        ax3.plot(range(0, self.batched_forces_plot[0].shape[1]), self.batched_forces_plot[1][frame,:,2], label='Target Force z')
        ax3.plot(range(0, self.batched_forces_plot[0].shape[1]), self.batched_forces_plot[0][frame,:,2], label='Trained Force z')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Force')
        ax3.set_title('Z Force Comparison')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()

    def plotABSForceMathing(self, frame = 0):

        plt.figure(figsize=(10, 5))

        # x-axis plot
        fig1, ax1 = plt.subplots()
        ax1.plot(range(0, self.batched_forces_plot[0].shape[1]), np.abs(self.batched_forces_plot[1][frame,:,0]), label='Target Force x')
        ax1.plot(range(0, self.batched_forces_plot[0].shape[1]), np.abs(self.batched_forces_plot[0][frame,:,0]), label='Trained Force x')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Force')
        ax1.set_title('X Force Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # y-axis plot
        fig2, ax2 = plt.subplots()
        ax2.plot(range(0, self.batched_forces_plot[0].shape[1]), np.abs(self.batched_forces_plot[1][frame,:,1]), label='Target Force y')
        ax2.plot(range(0, self.batched_forces_plot[0].shape[1]), np.abs(self.batched_forces_plot[0][frame,:,1]), label='Trained Force y')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Force')
        ax2.set_title('Y Force Comparison')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # z-axis plot
        fig1, ax3 = plt.subplots()
        ax3.plot(range(0, self.batched_forces_plot[0].shape[1]), np.abs(self.batched_forces_plot[1][frame,:,2]), label='Target Force z')
        ax3.plot(range(0, self.batched_forces_plot[0].shape[1]), np.abs(self.batched_forces_plot[0][frame,:,2]), label='Trained Force z')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Force')
        ax3.set_title('Z Force Comparison')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()

    def plotForceMagnitudeMatching(self, bead_index = 0, to_frame = 100):

        plt.figure(figsize=(10, 5))

        target_bead_forces  = np.linalg.norm(self.batched_forces_plot[1][:to_frame,bead_index,:], axis=-1)
        trained_bead_forces  = np.linalg.norm(self.batched_forces_plot[0][:to_frame,bead_index,:], axis=-1)


        plt.plot(range(0, to_frame), target_bead_forces, label='Target Force Mag')
        plt.plot(range(0, to_frame), trained_bead_forces, label='Train Force Mag')
        plt.xlabel('Epoch')
        plt.ylabel('Force kj mol^-1 nm^-1')
        plt.title('Target vs Trained Force magnitudes')
        # Place legend outside the box
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # plt.savefig("lossvsepochChignolinfoldednandunfolded177epochsearlystopped", format = "pdf")
        plt.show()

