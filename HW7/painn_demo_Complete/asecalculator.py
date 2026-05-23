import sys
import ase
import time

import torch
import numpy as np
import torch.cuda.amp as amp
import torch.optim as optim
# pyg
from torch_geometric.data import Data
# ase
from ase import io
from ase.data import atomic_numbers
from ase.calculators.calculator import Calculator, all_changes, all_properties
# self
from model import PaiNN 
from utils.pbc import build_pbc

class NNCalculator(Calculator):
    implemented_properties = ("energy", "forces", "stress", "magmoms")

    def __init__(self, model, device, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.step = 0
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        self.model_time = []
        self.ase_time = []

    def build_data(self, atoms):
        z = atoms.get_chemical_symbols()
        z = torch.tensor([atomic_numbers[element] for element in z], dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
        cell = torch.tensor(np.vstack(atoms.get_cell()), dtype=torch.float)
        shift, cell_num, i_num, j_num, pos_j = build_pbc(pos, cell, cutoff=self.model.cutoff)
        data = Data(
            z=z, pos=pos, cell=cell,
            shift=shift,
            cell_num=torch.tensor([cell_num, ], dtype=torch.long),
            i_num=torch.tensor([i_num, ], dtype=torch.long),
            j_num=torch.tensor([j_num, ], dtype=torch.long),
            pos_j=pos_j,
            batch=torch.zeros_like(z),
            idx=self.step,
        )
        self.step += 1
        data = data.to(self.device)
        return data

    def calculate(self, atoms, properties=all_properties, system_changes=all_changes):
        torch.cuda.synchronize()
        ase_start = time.time()
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        model_start = time.time()
        data = self.build_data(atoms)
        energy, force = self.model(data)
        torch.cuda.synchronize()
        model_end = time.time()
        model_time = model_end-model_start
        self.model_time.append(model_time)

        self.results.update(energy=energy.item(), free_energy=energy.item(), forces=force.detach_().cpu().numpy())
        ase_end = time.time()
        ase_time = ase_end - ase_start - model_time
        self.ase_time.append(ase_time)


def init_model(device):
    hidden_channel, num_layers, cutoff = 128, 3, 5.0
    sava_path = "best_model.pth"
    model = PaiNN(n_atom_basis=hidden_channel,n_interactions=num_layers)
    model.load_state_dict(torch.load(sava_path,weights_only=True))
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    atoms = io.read('./data/evaluate/96.cif', format='cif')

    model = init_model(device)

    calc = NNCalculator(model=model, device=device)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    force = atoms.get_forces()
    print("energy:",energy)
    print("force:\n",force)
