import os
import torch

from ase.io import Trajectory

import numpy as np
import time
# ase
from ase import io
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.andersen import Andersen
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
#
from asecalculator import init_model, NNCalculator
from utils.readtraj import readtraj

class MolecularDynamics:

    def __init__(
            self,
            atoms: Atoms,
            model: None,
            ensemble: str = "nvt",
            thermostat: str = "Berendsen_inhomogeneous",
            temperature: int = 300,
            timestep: float = 2.0,
            pressure: float = 1.01325e-4,
            taut=None,
            taup=None,
            friction: float = 1.0e-3,
            andersen_prob: float = 1.0e-2,
            ttime: float = 25.0,
            pfactor: float = 75.0 ** 2.0,
            external_stress=None,
            compressibility_au=None,
            trajectory=None,
            logfile=None,
            loginterval: int = 1,
            append_trajectory: bool = False,
            mask=None,
            device=None,
    ):

        self.atoms = atoms
        self.atoms.wrap()

        self.atoms.calc = NNCalculator(model=model, device=device)

        if taut is None: taut = 100 * timestep * units.fs
        if taup is None: taup = 1000 * timestep * units.fs
        if mask is None: mask = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        if external_stress is None: external_stress = 0.0

        if ensemble.lower() == "nve":
            self.dyn = VelocityVerlet(
                self.atoms,
                timestep * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
            print("NVE-MD created")

        elif ensemble.lower() == "nvt":
            self.upper_triangular_cell()
            print("time constant for temperature coupling in fs:", taut * units.fs)
            if thermostat.lower() == "nose-hoover":
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure * units.GPa,  # ase NPT does not like externalstress=None
                    ttime=taut * units.fs,
                    pfactor=None,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Nose-Hoover MD created")

            elif thermostat.lower().startswith("berendsen"):
                """
                Berendsen (constant N, V, T) molecular dynamics.
                """
                self.dyn = NVTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    taut=taut * units.fs,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )

        elif ensemble.lower() == "nvt_langevin":
            self.dyn = Langevin(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                friction=friction,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_andersen":
            self.dyn = Andersen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                andersen_prob=andersen_prob,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
        elif ensemble.lower() == "npt":
            """
            NPT ensemble default to Inhomogeneous_NPTBerendsen thermo/barostat
            This is a more flexible scheme that fixes three angles of the unit
            cell but allows three lattice parameter to change independently.
            """

            self.dyn = Inhomogeneous_NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                # append_trajectory=append_trajectory,
                # this option is not supported in ASE at this point (I have sent merge request there)
            )

        elif ensemble.lower() == "npt_berendsen":
            """

            This is a similar scheme to the Inhomogeneous_NPTBerendsen.
            This is a less flexible scheme that fixes the shape of the
            cell - three angles are fixed and the ratios between the three
            lattice constants.

            """

            self.dyn = NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt_nose_hoover":
            self.dyn = NPT(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                externalstress=external_stress,
                ttime=ttime * units.fs,
                pfactor=pfactor * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
                mask=mask,
            )

        else:
            raise ValueError("Ensemble not supported")

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

    def upper_triangular_cell(self, verbose=False):
        if not NPT._isuppertriangular(self.atoms.get_cell()):
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p ** 2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            self.atoms.set_cell(new_basis, scale_atoms=True)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)

    def run(self, steps: int):
        start = time.time()
        self.dyn.run(steps)
        end = time.time()
        print("ALL time",end-start)
        sum_model_time = sum(self.atoms.calc.model_time[5:])
        len_model_time = len(self.atoms.calc.model_time[5:])
        print("model sum",sum_model_time)
        print("model count",len_model_time)
        print("model avg",sum_model_time/len_model_time)
        sum_ase_time = sum(self.atoms.calc.ase_time[5:])
        len_ase_time = len(self.atoms.calc.ase_time[5:])
        print("ase sum",sum_ase_time)
        print("ase count",len_model_time)
        print("ase avg",sum_ase_time/len_ase_time)


if __name__ == "__main__":
    os.makedirs('./log', exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start = time.time()
    atoms = io.read(f'./data/evaluate/96.cif', format='cif')
    temperature=350
    MaxwellBoltzmannDistribution(atoms, temperature * units.kB)
    model = init_model(device)
    #print(model)

    md = MolecularDynamics(
            atoms=atoms,
            model=model,
            ensemble="nve",
            temperature=temperature,  # in K
            timestep=1,  # in femto-seconds
            taut=10,
            trajectory="./log/md.traj",
            logfile="./log/md.log",
            loginterval=20,
            device=device,
        )

    md.run(1500)  # run a step*timestep fs MD simulation
    print("consume_time:", time.time() - start)
    readtraj('./log/md.traj', './log/md.arc')


