# geodiff_numpy.py
import numpy as np
from math import sqrt

# ---------------------------
# Simple NumPy MLP (forward-only)
# ---------------------------
class MLP:
    def __init__(self, sizes, act=np.tanh, seed=0):
        """
        sizes: list of layer sizes, e.g. [in_dim, hid, hid, out_dim]
        act: activation function (elementwise)
        """
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            # Xavier init
            w = rng.normal(0, np.sqrt(2.0/(a+b)), size=(a, b)).astype(np.float64)
            bvec = np.zeros(b, dtype=np.float64)
            self.weights.append(w)
            self.biases.append(bvec)
        self.act = act

    def __call__(self, x):
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = self.act(h)
        return h

# ---------------------------
# Small equivariant-ish message passing block (forward-only)
# ---------------------------
class SimpleGeoMP:
    """
    A tiny message-passing block that:
      - uses relative vectors r_ij and distances
      - computes scalar messages via MLP on (h_i, h_j, dist)
      - aggregates messages and forms node-wise features
      - predicts noise vector per node (in 3D) using direction-weighting:
         predicted_eps_i = sum_j (direction_ij * scalar_ij)
    This is NOT a rigorous EGNN but preserves equivariant-like directional updates.
    """
    def __init__(self, node_feat_dim, edge_mlp_hidden=64, out_hidden=64, seed=1):
        self.edge_mlp = MLP([node_feat_dim*2 + 1, edge_mlp_hidden, edge_mlp_hidden], seed=seed)
        self.node_mlp = MLP([node_feat_dim + edge_mlp_hidden, out_hidden, out_hidden], seed=seed+1)
        # final heads:
        self.eps_head = MLP([out_hidden, out_hidden, 3], seed=seed+2)  # predicts vector (3)
        # For scalar weighting along direction:
        self.scalar_head = MLP([edge_mlp_hidden, edge_mlp_hidden, 1], seed=seed+3)

    def predict_eps(self, node_feats, coords, edge_index):
        """
        node_feats: (N, F)
        coords: (N, 3)
        edge_index: list of pairs (i,j) as shape (E,2)
        returns: eps_pred (N,3)
        """
        N = node_feats.shape[0]
        E = edge_index.shape[0]
        # compute relative vectors and distances
        r_ij = np.zeros((E, 3), dtype=np.float64)
        dist = np.zeros((E, 1), dtype=np.float64)
        # prepare edge input features
        edge_inputs = np.zeros((E, node_feats.shape[1]*2 + 1), dtype=np.float64)
        for e in range(E):
            i, j = edge_index[e]
            rv = coords[j] - coords[i]
            r_ij[e] = rv
            d = np.linalg.norm(rv) + 1e-12
            dist[e, 0] = d
            edge_inputs[e, :node_feats.shape[1]] = node_feats[i]
            edge_inputs[e, node_feats.shape[1]:2*node_feats.shape[1]] = node_feats[j]
            edge_inputs[e, -1] = d

        e_feat = self.edge_mlp(edge_inputs)  # (E, Ef)
        # scalar weight per edge
        scalar = self.scalar_head(e_feat).reshape(E, 1)  # (E,1)

        # direction normalized
        dir_ij = r_ij / (np.linalg.norm(r_ij, axis=1, keepdims=True) + 1e-12)  # (E,3)

        # contribution to node i: sum over edges where edge (i,j)
        eps_accum = np.zeros((N, 3), dtype=np.float64)
        # also build aggregated edge features per node for node update
        agg_edge_feat = np.zeros((N, e_feat.shape[1]), dtype=np.float64)

        for e in range(E):
            i, j = edge_index[e]
            weight = scalar[e, 0]
            # We choose orientation: message contributes to node i (from j)
            eps_accum[i] += (weight * dir_ij[e])
            agg_edge_feat[i] += e_feat[e]

        # node update
        node_input = np.concatenate([node_feats, agg_edge_feat], axis=1)  # (N, F+Ef)
        node_h = self.node_mlp(node_input)  # (N, H)
        eps_correction = self.eps_head(node_h)  # (N, 3)
        eps_pred = eps_accum + eps_correction  # combine directional and learned residual
        return eps_pred

# ---------------------------
# Diffusion scheduler utilities
# ---------------------------
def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T, dtype=np.float64)  # linear

def precompute_alphas(betas):
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    sqrt_alpha_bars = np.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = np.sqrt(1 - alpha_bars)
    return alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars

# ---------------------------
# Forward q(x_t | x_0) sampling (single molecule)
# ---------------------------
def q_sample(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, rng):
    """
    x0: (N,3)
    t: integer in [0, T-1]
    returns xt, eps (random noise used)
    """
    eps = rng.normal(size=x0.shape).astype(np.float64)
    xt = sqrt_alpha_bars[t] * x0 + sqrt_one_minus_alpha_bars[t] * eps
    return xt, eps

# ---------------------------
# DDPM reverse step (single molecule) using predicted eps
# ---------------------------
def p_sample(x_t, t, betas, alphas, alpha_bars, eps_pred, rng):
    """
    x_t: (N,3)
    t: current timestep (0..T-1)
    eps_pred: predicted noise by model for timestep t (N,3)
    returns x_{t-1}
    """
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    # follow DDPM formula:
    # mean = 1/sqrt(alpha_t) * ( x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_pred )
    coef = beta_t / (np.sqrt(1.0 - alpha_bar_t) + 1e-12)
    mean = (1.0 / np.sqrt(alpha_t)) * (x_t - coef * eps_pred)

    if t > 0:
        sigma = np.sqrt(beta_t)
        z = rng.normal(size=x_t.shape).astype(np.float64)
        x_prev = mean + sigma * z
    else:
        x_prev = mean  # at t=0 deterministic
    return x_prev

# ---------------------------
# High-level GeoDiff sampling for a batch (list of molecules)
# ---------------------------
class GeoDiffNumPy:
    def __init__(self, node_feat_dim, T=50, seed=0):
        self.T = T
        self.betas = make_beta_schedule(T)
        self.alphas, self.alpha_bars, self.sqrt_alpha_bars, self.sqrt_one_minus_alpha_bars = precompute_alphas(self.betas)
        self.model = SimpleGeoMP(node_feat_dim)
        self.rng = np.random.default_rng(seed)

    def sample_from_x0_batch(self, batch_node_feats, batch_coords, batch_edge_index):
        """
        Given lists for a batch of molecules, do forward q(x_t) -> pick some t (or highest t) and then reverse to produce denoised samples.
        Here we will:
          - for each molecule: sample xt at t = T-1 (max noise)
          - run reverse loop from T-1 downto 0, using model.predict_eps at each step
        Returns: batch of generated coords (list of arrays)
        """
        B = len(batch_coords)
        # forward: get x_T-1 noise for each molecule
        x_t_list = []
        eps_true_list = []
        for i in range(B):
            x0 = batch_coords[i]
            xt, eps = q_sample(x0, self.T-1, self.sqrt_alpha_bars, self.sqrt_one_minus_alpha_bars, self.rng)
            x_t_list.append(xt)
            eps_true_list.append(eps)

        # reverse sampling
        x_prev_list = x_t_list  # start from noisy
        for t in reversed(range(self.T)):
            new_list = []
            for i in range(B):
                x_t = x_prev_list[i]
                node_feats = batch_node_feats[i]
                edge_idx = batch_edge_index[i]  # shape (E,2)
                # predict eps
                eps_pred = self.model.predict_eps(node_feats, x_t, edge_idx)
                # step
                x_prev = p_sample(x_t, t, self.betas, self.alphas, self.alpha_bars, eps_pred, self.rng)
                new_list.append(x_prev)
            x_prev_list = new_list
        # after loop x_prev_list contains x_0_hat
        return x_prev_list

# ---------------------------
# Minimal demo / example usage
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Example batch with variable node counts (two molecules)
    # Molecule A: small chain (N=5)
    N1 = 5
    node_feats1 = np.eye(5)[:, :3].astype(np.float64)  # toy one-hot-like features (N, F=3)
    coords1 = rng.normal(scale=0.5, size=(N1, 3)).astype(np.float64)
    # edges: simple chain 0-1-2-3-4 (undirected as pairs)
    edges1 = np.array([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]])

    # Molecule B: smaller (N=3)
    N2 = 3
    node_feats2 = np.eye(3)[:, :3].astype(np.float64)
    coords2 = rng.normal(scale=0.2, size=(N2, 3)).astype(np.float64)
    edges2 = np.array([[0,1],[1,0],[1,2],[2,1]])

    batch_node_feats = [node_feats1, node_feats2]
    batch_coords = [coords1, coords2]
    batch_edge_index = [edges1, edges2]

    print("Original coords 1:\n", coords1)
    print("Original coords 2:\n", coords2)

    # create geodiff
    geodiff = GeoDiffNumPy(node_feat_dim=3, T=30, seed=123)

    # sample (forward->reverse)
    generated = geodiff.sample_from_x0_batch(batch_node_feats, batch_coords, batch_edge_index)

    print("\nGenerated coords 1 (denoised):\n", generated[0])
    print("\nGenerated coords 2 (denoised):\n", generated[1])

