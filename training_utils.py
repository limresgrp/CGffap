import os
from typing import Optional
import torch
import numpy as np

import training_modules as tm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset , DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

class CGDataset(Dataset):
    def __init__(self, npzdataset, stride = 1):
        super().__init__()
        self.data = {}
        for k, v in dict(npzdataset).items():
            if k in ['bead_pos', 'bead_forces']: #,'bead_types', 'bead_mass'
                self.data[k] = torch.from_numpy(v[::stride])
                self.data[k] = torch.from_numpy(v[::stride])

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
    def __init__(
        self,
        dataset,
        conf_bonds,
        conf_angles,
        conf_dihedrals,
        conf_bead_charges,
        model_weights = None,
        pos2unit = 1.,
        eng2unit = 1.,
        num_bead_types = None,
    ):
        super().__init__()
        potential = tm.ForceModel(
            dataset,
            conf_bonds,
            conf_angles,
            conf_dihedrals,
            conf_bead_charges,
            pos2unit,
            eng2unit,
            num_bead_types,
        )
        if model_weights is not None:
            try:
                potential.load_state_dict(torch.load(model_weights))
                print(f"Model weights {model_weights} successfully loaded!")
            except:
                print(f"Model weights {model_weights} are missing. Initializing default values.")
        self.model  = tm.ForceMapper(potential)

        self.train_pos, self.valid_pos, self.train_force , self.valid_force = train_test_split(dataset['bead_pos'], dataset['bead_forces'], test_size=0.10, random_state=42)

        self.loss_matrix = []
        self.per_frame = []  
        self.train_loss = None
        self.valid_loss = None
        self.zero = None
        self.losswith0 = []
        self.batched_forces_plot = []
        self.batched_val_forces_plot = []
        self.initialguess = []

    def initiateTraining(
        self,
        epochs: int = 1000,
        batch_size: int = 100,
        lr: float = 1.e-3,
        num_workers: int = 0,
        patience: int = 100,
        checkpoint_every: Optional[int] = None,
        model_name = 'model',
        device: str = 'cuda',
    ):
        self.model.to(device)
        writer = SummaryWriter()
        
        charge_param_names = [
            'model.module.bead_charges_vals',
            'model.module.f_r',
        ]
        charge_params = [p for name, p in self.named_parameters() if name in charge_param_names]

        constrained_param_names = [
            'model.module.angle_spring_constant_vals', 
            'model.module.improper_dihedral_const_vals', 
            'model.module.dispertion_const',
            'model.module.bead_radii',
        ]
        constrained_params = [p for name, p in self.named_parameters() if name in constrained_param_names]

        base_params = [
            p for name, p in self.named_parameters() if name not in (
                charge_param_names + constrained_param_names
            )
        ]

        optimizer =  Adam([
                {'params': constrained_params, 'lr': 5e-3, 'weight_decay': 0},
                {'params': charge_params, 'lr': 1e-5, 'weight_decay': 0},
                {'params': base_params}
            ],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
        
        loss_fn = MSELoss() # L1Loss() #    LBFGS(self.model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None) #
        
        dataset_train = {
            'bead_pos': self.train_pos,
            'bead_forces': self.train_force
        }

        loader = DataLoader(
            CGDataset(dataset_train, stride=1),
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle=True,
        )

        print(f"Training dataset loaded: {len(self.train_pos)} training samples")

        dataset_val = {
            'bead_pos': self.valid_pos,
            'bead_forces': self.valid_force
        }

        val_loader = DataLoader(
            CGDataset(dataset_val),
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle=False,
        )

        print(f"Validation dataset loaded: {len(self.valid_pos)} validation samples")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')  # Initialize with a large value
        patience = patience  # Number of ep
        patience_counter = 0

        mean_zeroloss = loss_fn(torch.Tensor(dataset_train['bead_forces'])*0, torch.Tensor(dataset_train['bead_forces']) ).detach().cpu().numpy()
        mean_zeroloss_val = loss_fn(torch.Tensor(dataset_val['bead_forces'])*0, torch.Tensor(dataset_val['bead_forces']) ).detach().cpu().numpy()


        for m_epoch in range(1, epochs + 1):
            if checkpoint_every is not None and (m_epoch % checkpoint_every == 0):
                os.makedirs('models/', exist_ok=True)
                torch.save(self.model.module.state_dict(), 'models/' + model_name + f'{m_epoch}.pth')
                torch.jit.script(self.model.module).save('models/' + model_name + f'{m_epoch}.pt')

            loss_plot = []

             # Validation
            self.model.eval()
            val_loss_plot = []
            for val_batch in val_loader:
                for k, v in val_batch.items():
                    val_batch[k] = v.to(device)
                for k, v in val_batch.items():
                    val_batch[k] = v.to(device)
                val_out = self.model(val_batch['bead_pos'])
                val_loss = loss_fn(val_out , val_batch['bead_forces'] )
                self.batched_val_forces_plot = [val_out.detach().cpu().numpy() , val_batch['bead_forces'].detach().cpu().numpy()]
                val_loss_plot.append(val_loss.detach().cpu().numpy())

            # Training
            self.model.train()
            for batch in loader:
                # for epoch in range(0,10):
                # for batch in loader:
                for k, v in batch.items():
                    batch[k] = v.to(device)

                optimizer.zero_grad()
                out = self.model(batch['bead_pos'])
                # print(out[0])
                # print(out[0] * 41.84)
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
                    if n in ['module.equ_val_angles_vals']:
                        p.data = p.data.clamp_(min=0, max = torch.pi)
                    if n in ['module.proper_phase_shift', 'module.proper_phase_shift_BB']:
                        p.data = p.data.clamp_(min=-torch.pi, max = torch.pi)
                    if n in ['module.bead_radii']:
                        p.data = p.data.clamp_(min=1.2)
                    # self.per_frame.append(loss_plot)
                # self.losswith0.append(losswith0single)
            # loss_matrix.append(np.mean(loss_plot))
            # print("loss = ",np.mean(loss_plot), "epoch =", m_epoch)

            # Print and save results
            mean_train_loss = np.mean(loss_plot)
            mean_val_loss = np.mean(val_loss_plot)
            print(f"Epoch {m_epoch}: Train Loss = {mean_train_loss:.3f}, Val Loss = {mean_val_loss:.3f}, Zero Loss Train = {mean_zeroloss:.3f}, Zero Loss Val = {mean_zeroloss_val:.3f}")

            # Store losses for plotting
            train_losses.append(mean_train_loss)
            val_losses.append(mean_val_loss)

            # Save model checkpoint if validation loss improves
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                os.makedirs('models/', exist_ok=True)
                print('Best model!')
                torch.save(self.model.module.state_dict(), 'models/' + model_name + '.best.pth')
                torch.jit.script(self.model.module).save('models/' + model_name +'.best.pt')
                self.batched_forces_plot = [out.detach().cpu().numpy()  , batch['bead_forces'].detach().cpu().numpy()]
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            # Check for early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {m_epoch} epochs.")
                break

        writer.flush()
        self.train_loss = np.array(train_losses).reshape(-1)
        self.valid_loss = np.array(val_losses).reshape(-1)
            
        return self.model
    
    def plotLosses(self, truncate = 0):
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, len(self.train_loss[truncate:])), self.train_loss[truncate:], label='Training Loss')
        plt.plot(range(0, len(self.valid_loss[truncate:])), self.valid_loss[truncate:], label='Validation Loss')
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

