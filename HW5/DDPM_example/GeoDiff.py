import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# ---------------------------------------------------------------------------
# MLP (nn.Module)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, sizes, act=nn.Tanh):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Equivariant-ish message-passing block (nn.Module, fully vectorized)
# ---------------------------------------------------------------------------
class SimpleGeoMP(nn.Module):
    """
    Tiny message-passing block:
      - relative vectors r_ij and distances as edge features
      - scalar messages via MLP on (h_i, h_j, dist)
      - directional aggregation: eps_i = sum_j (scalar_ij * dir_ij)
      - node MLP residual on top
    All edge loops replaced by torch.scatter_add for autograd compatibility.
    """

    def __init__(self, node_feat_dim, edge_mlp_hidden=64, out_hidden=64):
        super().__init__()
        self.edge_mlp = MLP([node_feat_dim * 2 + 1, edge_mlp_hidden, edge_mlp_hidden])
        self.node_mlp = MLP([node_feat_dim + edge_mlp_hidden, out_hidden, out_hidden])
        self.eps_head = MLP([out_hidden, out_hidden, 3])
        self.scalar_head = MLP([edge_mlp_hidden, edge_mlp_hidden, 1])

    def forward(self, node_feats, coords, edge_index):
        """
        node_feats : (N, F)  float tensor
        coords     : (N, 3)  float tensor
        edge_index : (E, 2)  long tensor  — each row is (src, dst)
        returns      (N, 3)  predicted noise
        """
        src = edge_index[:, 0]   # (E,)
        dst = edge_index[:, 1]   # (E,)
        N = node_feats.shape[0]

        rv = coords[dst] - coords[src]                        # (E, 3)
        dist = rv.norm(dim=1, keepdim=True).clamp(min=1e-12)  # (E, 1)
        dir_ij = rv / dist                                     # (E, 3) unit vectors

        edge_inputs = torch.cat([node_feats[src], node_feats[dst], dist], dim=1)
        e_feat = self.edge_mlp(edge_inputs)   # (E, Ef)
        scalar = self.scalar_head(e_feat)     # (E, 1)

        # weighted directional contribution per edge → aggregate to source node
        contrib = scalar * dir_ij             # (E, 3)
        idx3 = src.unsqueeze(1).expand(-1, 3)
        eps_accum = torch.zeros(N, 3, device=coords.device, dtype=coords.dtype)
        eps_accum.scatter_add_(0, idx3, contrib)

        # aggregate edge features for node update
        Ef = e_feat.shape[1]
        idxEf = src.unsqueeze(1).expand(-1, Ef)
        agg_edge_feat = torch.zeros(N, Ef, device=coords.device, dtype=coords.dtype)
        agg_edge_feat.scatter_add_(0, idxEf, e_feat)

        node_input = torch.cat([node_feats, agg_edge_feat], dim=1)
        node_h = self.node_mlp(node_input)
        eps_correction = self.eps_head(node_h)   # (N, 3)

        return eps_accum + eps_correction


# ---------------------------------------------------------------------------
# Diffusion schedule helpers
# ---------------------------------------------------------------------------
def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def precompute_alphas(betas):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = alpha_bars.sqrt()
    sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()
    return alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars


# ---------------------------------------------------------------------------
# Forward process  q(x_t | x_0)
# ---------------------------------------------------------------------------
def q_sample(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars):
    """
    x0 : (N, 3)
    t  : int in [0, T-1]
    Returns xt (N, 3) and eps (N, 3).
    """
    eps = torch.randn_like(x0)
    xt = sqrt_alpha_bars[t] * x0 + sqrt_one_minus_alpha_bars[t] * eps
    return xt, eps


# ---------------------------------------------------------------------------
# Loss function: DDPM noise-prediction MSE
# ---------------------------------------------------------------------------
def diffusion_loss(model, node_feats, coords, edge_index,
                   t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars):
    """
    Samples x_t from x_0 at timestep t and computes
        L = MSE( model(x_t, t), eps )
    where eps is the noise added in q(x_t | x_0).
    """
    xt, eps_true = q_sample(coords, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)
    eps_pred = model(node_feats, xt, edge_index)
    return F.mse_loss(eps_pred, eps_true)


# ---------------------------------------------------------------------------
# DDPM reverse step  p(x_{t-1} | x_t)
# ---------------------------------------------------------------------------
@torch.no_grad()
def p_sample(x_t, t, betas, alphas, alpha_bars, eps_pred):
    """
    x_t      : (N, 3)
    eps_pred : (N, 3) model output
    Returns x_{t-1} (N, 3).
    """
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    coef = beta_t / (1.0 - alpha_bar_t).sqrt().clamp(min=1e-12)
    mean = (1.0 / alpha_t.sqrt()) * (x_t - coef * eps_pred)

    if t > 0:
        z = torch.randn_like(x_t)
        x_prev = mean + beta_t.sqrt() * z
    else:
        x_prev = mean   # deterministic final step
    return x_prev


# ---------------------------------------------------------------------------
# GeoDiff: wraps model, schedule, optimizer, training, and sampling
# ---------------------------------------------------------------------------
class GeoDiff:
    def __init__(self, node_feat_dim, T=50, lr=1e-3, device=None):
        self.T = T
        self.device = device or torch.device("cpu")

        betas = make_beta_schedule(T).to(self.device)
        alphas, alpha_bars, sqrt_ab, sqrt_one_minus_ab = precompute_alphas(betas)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = sqrt_ab
        self.sqrt_one_minus_alpha_bars = sqrt_one_minus_ab

        self.model = SimpleGeoMP(node_feat_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    def _to_tensor(self, arr, dtype=torch.float32):
        if isinstance(arr, torch.Tensor):
            return arr.to(device=self.device, dtype=dtype)
        return torch.tensor(arr, dtype=dtype, device=self.device)

    # ------------------------------------------------------------------
    def train_step(self, batch_node_feats, batch_coords, batch_edge_index):
        """
        One gradient update on the batch.

        Args:
            batch_node_feats  : list of (Ni, F) array/tensor per molecule
            batch_coords      : list of (Ni, 3) array/tensor
            batch_edge_index  : list of (Ei, 2) int array/tensor

        Returns:
            float  average loss over the batch
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = torch.zeros(1, device=self.device)
        for nf, co, ei in zip(batch_node_feats, batch_coords, batch_edge_index):
            nf = self._to_tensor(nf, torch.float32)
            co = self._to_tensor(co, torch.float32)
            ei = self._to_tensor(ei, torch.long)

            t = torch.randint(0, self.T, (1,)).item()
            loss = diffusion_loss(self.model, nf, co, ei, t,
                                  self.sqrt_alpha_bars,
                                  self.sqrt_one_minus_alpha_bars)
            total_loss = total_loss + loss

        total_loss = total_loss / len(batch_coords)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_from_x0_batch(self, batch_node_feats, batch_coords, batch_edge_index):
        """
        Forward q(x_{T-1}) for each molecule, then reverse-denoise from T-1 → 0.

        Args:
            batch_node_feats  : list of (Ni, F) arrays/tensors
            batch_coords      : list of (Ni, 3) arrays/tensors  (x_0 seeds)
            batch_edge_index  : list of (Ei, 2) int arrays/tensors

        Returns:
            list of (Ni, 3) tensors  — denoised coordinates
        """
        self.model.eval()

        # --- forward: start from x_{T-1} ---
        x_t_list = []
        for co in batch_coords:
            co = self._to_tensor(co, torch.float32)
            xt, _ = q_sample(co, self.T - 1,
                              self.sqrt_alpha_bars,
                              self.sqrt_one_minus_alpha_bars)
            x_t_list.append(xt)

        # --- reverse: iterate t from T-1 down to 0 ---
        for t in reversed(range(self.T)):
            new_list = []
            for i in range(len(batch_coords)):
                x_t = x_t_list[i]
                nf = self._to_tensor(batch_node_feats[i], torch.float32)
                ei = self._to_tensor(batch_edge_index[i], torch.long)
                eps_pred = self.model(nf, x_t, ei)
                x_prev = p_sample(x_t, t,
                                  self.betas, self.alphas, self.alpha_bars,
                                  eps_pred)
                new_list.append(x_prev)
            x_t_list = new_list

        return x_t_list   # x_0 estimates


# ---------------------------------------------------------------------------
# Demo / example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Molecule A: 5 atoms
    N1 = 5
    node_feats1 = np.eye(5)[:, :3].astype(np.float32)
    coords1 = rng.normal(scale=0.5, size=(N1, 3)).astype(np.float32)
    edges1 = np.array([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]])

    # Molecule B: 3 atoms
    N2 = 3
    node_feats2 = np.eye(3)[:, :3].astype(np.float32)
    coords2 = rng.normal(scale=0.2, size=(N2, 3)).astype(np.float32)
    edges2 = np.array([[0,1],[1,0],[1,2],[2,1]])

    batch_node_feats = [node_feats1, node_feats2]
    batch_coords     = [coords1, coords2]
    batch_edge_index = [edges1, edges2]

    print("Original coords 1:\n", coords1)
    print("Original coords 2:\n", coords2)

    geodiff = GeoDiff(node_feat_dim=3, T=30, lr=1e-3)

    # --- training loop ---
    training_steps = 50
    print(f"\n--- Training {training_steps} steps---")
    for step in range(training_steps):
        loss = geodiff.train_step(batch_node_feats, batch_coords, batch_edge_index)
        print(f"  step {step+1:3d}  loss={loss:.6f}")

    # --- sampling ---
    print("\n--- Sampling ---")
    generated = geodiff.sample_from_x0_batch(batch_node_feats, batch_coords, batch_edge_index)
    print("\nGenerated coords 1 (denoised):\n", generated[0].cpu().numpy())
    print("\nGenerated coords 2 (denoised):\n", generated[1].cpu().numpy())
