
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional
from torch.autograd import grad
# from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from model_block import PaiNNInteraction,PaiNNMixing,CosineCutoff,GaussianRBF
from utils.common import radius_graph


class PaiNN(nn.Module):
    def __init__(
            self,
            n_atom_basis: int,
            n_interactions: int,
            max_z: int = 100,
            shared_filters: bool = False,
            epsilon: float = 1e-8,
    ):
        super(PaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions

        cutoff_fn = CosineCutoff(5.0)
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = GaussianRBF(n_rbf=20, cutoff=5.0)

        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)  # 100 128
        self.E_emb = nn.Embedding(max_z,1)

        self.share_filters = shared_filters
        self.filter_net = nn.Linear(self.radial_basis.n_rbf,self.n_interactions * n_atom_basis * 3)
        self.interactions = nn.ModuleList()
        for i in range(self.n_interactions):
            self.interactions.append(PaiNNInteraction(n_atom_basis=self.n_atom_basis))
        self.mixing = nn.ModuleList()
        for i in range(self.n_interactions):
            self.mixing.append(PaiNNMixing( n_atom_basis=self.n_atom_basis, epsilon=epsilon))

        self.outnet = nn.Sequential(
            nn.Linear(n_atom_basis,n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis,n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 1),
        )

    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        pos.requires_grad_(True)
        edge_index, shift, _j = radius_graph(data,self.cutoff)
        j, i = edge_index
        v_r = (pos[j] - pos[i] + shift)

        atomic_numbers = z  # 每个原子的Z
        r_ij = v_r  # j-i 矢量
        idx_i = i  # i
        idx_j = j  # j
        n_atoms = atomic_numbers.shape[0] 
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)

        phi_ij = self.radial_basis(d_ij) 
        fcut = self.cutoff_fn(d_ij)[..., None] 
        filters = self.filter_net(phi_ij) * fcut
        filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)  

        dir_ij = r_ij / d_ij  
        q = self.embedding(atomic_numbers)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
        q = q.squeeze(1)
        y = self.outnet(q) + self.E_emb(z)
        out = scatter(y, batch, dim=0)
        grad_outputs = [torch.ones_like(out)]
        dy = grad([out], [pos], grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
        return out, -dy


if __name__ == '__main__':
    pass
