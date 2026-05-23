import torch
import math
from torch import nn

def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut

class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y
class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)
#############################################################################################################
@torch.jit.script
def _scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y

def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    return _scatter_add(x, idx_i, dim_size, dim)
#############################################################################################################
class PaiNNInteraction(nn.Module):

    def __init__(self, n_atom_basis: int):
        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.interatomic_context_net = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 3 * n_atom_basis),
        )

    def forward(self, q, mu, Wij, dir_ij, idx_i, idx_j, n_atoms):
        # inter-atomic
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)  
        dq = scatter_add(dq, idx_i, dim_size=n_atoms)  

        muj = mu[idx_j]
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms)  

        q = q + dq 
        mu = mu + dmu  
        return q, mu


class PaiNNMixing(nn.Module):

    def __init__(self, n_atom_basis, epsilon):
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.intraatomic_context_net = nn.Sequential(
            nn.Linear(2 * n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 3 * n_atom_basis),
        )
        self.mu_channel_mix = nn.Linear(n_atom_basis, 2 * n_atom_basis ,bias=False)
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        ## intra-atomic
        mu_mix = self.mu_channel_mix(mu) 

        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1) 
        mu_Vn = torch.sqrt(torch.sum(mu_V ** 2, dim=-2, keepdim=True) + self.epsilon) 

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)  

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W
        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu
