import os
import torch

from ase.io import Trajectory
from ase import Atoms
from ase.optimize import BFGS
from ase import io
from ase import Atoms, units

from asecalculator import init_model, NNCalculator
import numpy as np
from utils.readtraj import readtraj


if __name__ == "__main__":
    os.makedirs('./log', exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    atoms = io.read(f'./data/evaluate/96.cif', format='cif')
    model = init_model(device)
    calc = NNCalculator(model=model, device=device)
    atoms.calc = calc

    dyn = BFGS(atoms,trajectory='./log/opt.traj')
    dyn.run(fmax=0.01)
    readtraj('./log/opt.traj', './log/opt.arc')



