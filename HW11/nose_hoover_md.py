"""
============================================================
NVT Molecular Dynamics with Nose-Hoover Thermostat
Pair Potential: Lennard-Jones  (Reduced Units: sigma=1, eps=1, m=1, kB=1)
============================================================

Key Theoretical Formulas
-------------------------
1. Lennard-Jones potential:
   V(r) = 4ε [(σ/r)^12 - (σ/r)^6]

2. LJ pair force on atom i due to atom j:
   F_ij = (48ε/r²) [(σ/r)^12 - 0.5(σ/r)^6] * r_ij

3. Nose-Hoover equations of motion (extended system):
   dr_i/dt  = p_i / m_i
   dp_i/dt  = F_i  -  ξ · p_i           (friction term)
   dξ/dt    = (1/Q) · (2K - Nf·kB·T)    (thermostat EOM)
   dη/dt    = ξ                           (NH "position")

4. Thermostat mass:
   Q = Nf · kB · T · τ_NH²

5. NH conserved pseudo-Hamiltonian:
   H* = KE + PE + Q·ξ²/2 + Nf·kB·T·η

6. Temperature from kinetic energy:
   T = 2K / (Nf · kB)

7. Long-range tail correction (energy):
   U_tail = (8π·N·ρ/3) · [1/(3·rc^9) - 1/rc^3]

8. Pressure (virial theorem):
   P = ρ·kBT + W/(3V),   W = Σ_{i<j} r_ij · F_ij

9. Velocity Verlet (modified for NH friction):
   Step by Trotter splitting of Liouville operator
   (Martyna et al., Mol. Phys. 87, 1117, 1996)

Author: AI4Chem
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# Main class
# ======================================================================

class NoseHooverMD:
    """
    NVT Lennard-Jones MD with Nose-Hoover thermostat.

    Parameters
    ----------
    N        : number of atoms
    rho      : number density  (N/V, reduced units)
    T_target : target temperature  (kB = 1)
    dt       : integration time step
    tau_nh   : Nose-Hoover coupling time constant
    """

    def __init__(self, N: int = 108, rho: float = 0.8,
                 T_target: float = 1.0, dt: float = 0.002,
                 tau_nh: float = 0.5):

        self.N        = N
        self.T_target = T_target
        self.dt       = dt
        self.rho      = rho

        # Box length from ρ = N/V  →  L = (N/ρ)^(1/3)
        self.L = (N / rho) ** (1.0 / 3.0)

        # LJ cutoff radius (standard 2.5σ, but ≤ L/2)
        self.rc  = min(2.5, 0.49 * self.L)
        rc3      = self.rc ** 3
        rc9      = rc3 ** 3

        # Long-range tail correction to total energy (per atom):
        #   u_tail = (8π·ρ/3)·[1/(3·rc^9) - 1/rc^3]
        self.e_tail = (8.0 * np.pi * rho / 3.0) * (1.0 / (3.0 * rc9) - 1.0 / rc3)

        # Degrees of freedom: 3N - 3 (COM translation removed)
        self.ndof = 3 * N - 3

        # ---- Nose-Hoover thermostat variables ----
        # Thermostat "mass":  Q = Nf·kB·T·τ²
        self.Q   = self.ndof * T_target * tau_nh ** 2
        self.xi  = 0.0   # friction variable (dη/dt)
        self.eta = 0.0   # NH position variable

        # Arrays
        self.pos    = np.zeros((N, 3))
        self.vel    = np.zeros((N, 3))
        self.forces = np.zeros((N, 3))

        self._init_fcc_lattice()
        self._init_velocities()
        self.forces, self._pe, self._vir = self._compute_forces()

        print("=" * 55)
        print("  Nose-Hoover MD  –  LJ fluid (reduced units)")
        print("=" * 55)
        print(f"  N       = {N},   ρ = {rho:.3f},   L = {self.L:.4f}")
        print(f"  T_target= {T_target},  dt = {dt},  τ_NH = {tau_nh}")
        print(f"  Q       = {self.Q:.4f},  Nf = {self.ndof},  rc = {self.rc:.3f}")
        print("=" * 55)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_fcc_lattice(self):
        """Atoms on an FCC lattice (4-atom unit cell)."""
        basis = np.array([[0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.0],
                          [0.5, 0.0, 0.5],
                          [0.0, 0.5, 0.5]])
        nc = int(np.ceil((self.N / 4) ** (1.0 / 3.0)))
        a  = self.L / nc          # lattice constant

        coords = []
        for ix in range(nc):
            for iy in range(nc):
                for iz in range(nc):
                    for b in basis:
                        if len(coords) < self.N:
                            coords.append((np.array([ix, iy, iz]) + b) * a)
        self.pos = np.array(coords[:self.N])

    def _init_velocities(self):
        """Maxwell-Boltzmann distribution, then rescale to exact T_target."""
        np.random.seed(42)
        # Draw velocities: σ_v = sqrt(kB·T/m) = sqrt(T)  (kB=m=1)
        self.vel = np.random.randn(self.N, 3) * np.sqrt(self.T_target)
        # Remove centre-of-mass drift
        self.vel -= self.vel.mean(axis=0)
        # Rescale to exact T_target
        ke    = 0.5 * np.sum(self.vel ** 2)
        T_now = 2.0 * ke / self.ndof
        self.vel *= np.sqrt(self.T_target / T_now)

    # ------------------------------------------------------------------
    # LJ Force & Energy
    # ------------------------------------------------------------------

    def _compute_forces(self):
        """
        Compute LJ forces (vectorised inner loop), potential energy,
        and virial  W = Σ r_ij · F_ij.

        V(r)  = 4 [(1/r)^12 - (1/r)^6]           (σ=ε=1)
        F/r²  = 48 r^{-2}[(r^{-6})^2 - 0.5·r^{-6}]
        Cutoff-shifted: V_s(r) = V(r) - V(rc)
        """
        L    = self.L
        rc2  = self.rc ** 2
        N    = self.N
        pos  = self.pos

        # Energy shift so V_s(rc) = 0
        r2c   = 1.0 / rc2
        r6c   = r2c ** 3
        e_cut = 4.0 * r6c * (r6c - 1.0)

        forces = np.zeros((N, 3))
        pe     = 0.0
        virial = 0.0

        for i in range(N - 1):
            # Displacement vectors: r_i - r_j  for all j > i
            dr  = pos[i] - pos[i + 1:]          # shape (N-1-i, 3)
            dr -= L * np.round(dr / L)           # minimum image convention
            r2  = np.einsum('ij,ij->i', dr, dr)  # |dr|²

            mask = r2 < rc2
            if not np.any(mask):
                continue

            r2m  = r2[mask]
            drm  = dr[mask]

            r2i  = 1.0 / r2m
            r6i  = r2i ** 3   # (σ/r)^6  with σ=1

            # Force coefficient:  dV/dr · (1/r) = 48ε/r² [(σ/r)^12 - 0.5(σ/r)^6]
            f_r2 = 48.0 * r2i * r6i * (r6i - 0.5)
            fvec = f_r2[:, None] * drm            # force vectors on atom i

            forces[i] += fvec.sum(axis=0)

            j_idx = np.where(mask)[0] + (i + 1)
            np.add.at(forces, j_idx, -fvec)       # Newton's 3rd law

            pe     += np.sum(4.0 * r6i * (r6i - 1.0) - e_cut)
            virial += np.sum(f_r2 * r2m)          # Σ r·F

        pe += self.e_tail * N   # long-range correction
        return forces, pe, virial

    # ------------------------------------------------------------------
    # Nose-Hoover velocity-Verlet integrator
    # ------------------------------------------------------------------

    def _ke(self) -> float:
        """Kinetic energy  K = ½·m·Σv²  (m=1)."""
        return 0.5 * np.sum(self.vel ** 2)

    def step(self):
        """
        Single time step using the Nose-Hoover velocity-Verlet scheme
        (Trotter factorisation, Martyna et al. 1996).

        Integration order
        -----------------
        1.  ξ ← ξ + (dt/2)·G_ξ          (half-step thermostat)
        2.  v ← v·exp(−ξ·dt/2)           (velocity scaling, half)
        3.  v ← v + (dt/2)·F/m           (half-kick from forces)
        4.  r ← r + dt·v                 (full position update)
        5.  F ← F(r_new)                 (recompute forces)
        6.  v ← v + (dt/2)·F_new/m       (half-kick, new forces)
        7.  v ← v·exp(−ξ·dt/2)           (velocity scaling, half)
        8.  ξ ← ξ + (dt/2)·G_ξ          (half-step thermostat)
        9.  η ← η + dt·ξ                 (NH position, full step)

        where  G_ξ = (2K − Nf·kB·T) / Q
        """
        dt  = self.dt
        dt2 = 0.5 * dt

        # ---- 1. Half-step ξ ----
        G        = (2.0 * self._ke() - self.ndof * self.T_target) / self.Q
        self.xi += dt2 * G

        # ---- 2. Scale velocities (NH damping) ----
        scale     = np.exp(-self.xi * dt2)
        self.vel *= scale

        # ---- 3. Half velocity kick from current forces ----
        self.vel += dt2 * self.forces   # m = 1

        # ---- 4. Full position update with PBC ----
        self.pos  = (self.pos + dt * self.vel) % self.L

        # ---- 5. Recompute forces at new positions ----
        self.forces, pe, virial = self._compute_forces()

        # ---- 6. Half velocity kick from new forces ----
        self.vel += dt2 * self.forces

        # ---- 7. Scale velocities again ----
        self.vel *= scale   # same scale factor for time-reversibility

        # ---- 8. Half-step ξ with updated KE ----
        G        = (2.0 * self._ke() - self.ndof * self.T_target) / self.Q
        self.xi += dt2 * G

        # ---- 9. Full step η ----
        self.eta += dt * self.xi

        ke = self._ke()
        return ke, pe, virial

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self, n_steps: int = 20000,
            equil_steps: int = 5000,
            output_every: int = 500):
        """
        Run the simulation.

        Parameters
        ----------
        n_steps      : total number of MD steps
        equil_steps  : equilibration steps (not collected in data)
        output_every : print / record interval
        """
        hdr = (f"{'Step':>8}  {'T':>8}  {'KE':>10}  "
               f"{'PE':>12}  {'E_tot':>12}  {'ξ':>9}  {'P':>8}")
        print("\n" + hdr)
        print("-" * len(hdr))

        data = {k: [] for k in ('t', 'T', 'KE', 'PE', 'E', 'xi', 'P', 'H_star')}

        for step in range(n_steps):
            ke, pe, virial = self.step()

            T      = 2.0 * ke / self.ndof
            P      = self.rho * T + virial / (3.0 * self.L ** 3)
            E      = ke + pe
            # NH conserved quantity: H* = KE + PE + Q·ξ²/2 + Nf·kBT·η
            H_star = E + 0.5 * self.Q * self.xi ** 2 + self.ndof * self.T_target * self.eta

            if step % output_every == 0:
                print(f"{step:>8}  {T:>8.4f}  {ke:>10.4f}  "
                      f"{pe:>12.4f}  {E:>12.4f}  {self.xi:>9.5f}  {P:>8.4f}")

            if step >= equil_steps:
                data['t'].append(step * self.dt)
                data['T'].append(T)
                data['KE'].append(ke)
                data['PE'].append(pe)
                data['E'].append(E)
                data['xi'].append(self.xi)
                data['P'].append(P)
                data['H_star'].append(H_star)

        for k in data:
            data[k] = np.array(data[k])

        self._print_summary(data)
        return data

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _print_summary(self, data):
        print(f"\n{'='*50}")
        print("  Production averages")
        print(f"{'='*50}")
        print(f"  T_target = {self.T_target:.4f}")
        print(f"  <T>      = {data['T'].mean():.4f} ± {data['T'].std():.4f}")
        print(f"  <KE>     = {data['KE'].mean():.4f}")
        print(f"  <PE>     = {data['PE'].mean():.4f}")
        print(f"  <E_tot>  = {data['E'].mean():.4f}")
        print(f"  <P>      = {data['P'].mean():.4f}")
        H0  = data['H_star'][0]
        dH  = (data['H_star'] - H0) / abs(H0)
        print(f"  ΔH*/H*   = {dH.std():.2e}  (NH conserved quantity drift)")
        print(f"{'='*50}")

    def plot(self, data, save: str = "nose_hoover_md.png"):
        """Four-panel diagnostic plot."""
        t = data['t']

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Nose-Hoover MD — LJ fluid (reduced units)", fontsize=13, fontweight='bold')

        # --- (0,0) Temperature ---
        ax = axes[0, 0]
        ax.plot(t, data['T'], lw=0.8, color='steelblue', label='T(t)')
        ax.axhline(self.T_target, color='red', ls='--', lw=1.5,
                   label=f'Target T = {self.T_target}')
        ax.axhline(np.mean(data['T']), color='green', ls=':', lw=1.5,
                   label=f"<T> = {np.mean(data['T']):.4f}")
        ax.set(xlabel='Time (τ)', ylabel='T (reduced)', title='Temperature')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # --- (0,1) Energy components ---
        ax = axes[0, 1]
        ax.plot(t, data['E'],  lw=0.8, color='black',      label='E_tot')
        ax.plot(t, data['KE'], lw=0.8, color='tomato',     alpha=0.7, label='KE')
        ax.plot(t, data['PE'], lw=0.8, color='dodgerblue', alpha=0.7, label='PE')
        ax.set(xlabel='Time (τ)', ylabel='Energy (reduced)', title='Energy Components')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # --- (1,0) Nose-Hoover friction variable ξ ---
        ax = axes[1, 0]
        ax.plot(t, data['xi'], lw=0.8, color='purple')
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set(xlabel='Time (τ)', ylabel='ξ (friction)',
               title='NH Friction Variable ξ(t)')
        ax.grid(alpha=0.3)

        # --- (1,1) NH Conserved Quantity H* ---
        ax = axes[1, 1]
        H0   = data['H_star'][0]
        dH   = (data['H_star'] - H0) / abs(H0)
        ax.plot(t, dH, lw=0.8, color='darkorange')
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set(xlabel='Time (τ)',
               ylabel='ΔH*/|H*₀|',
               title='NH Conserved Quantity Drift')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved → {save}")
        plt.show()


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    md = NoseHooverMD(
        N        = 108,    # atoms (4 × 3³ FCC unit cells)
        rho      = 0.8,    # number density
        T_target = 1.0,    # target temperature (kB = 1)
        dt       = 0.002,  # time step (LJ reduced units)
        tau_nh   = 0.5,    # NH coupling time constant
    )

    data = md.run(
        n_steps     = 20000,   # total MD steps  (= 40 τ)
        equil_steps = 5000,    # equilibration   (= 10 τ)
        output_every= 500,
    )

    md.plot(data)
