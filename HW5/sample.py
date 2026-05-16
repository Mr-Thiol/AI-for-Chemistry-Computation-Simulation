import os
import re
import torch
import pickle
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

from torch_geometric.data import Batch

from models.epsnet import get_model
from utils.datasets import PackedConformationDataset
from utils.misc import get_logger, get_new_log_dir, seed_all
from utils.transforms import Compose, CountNodesPerGraph, AddHigherOrderEdges


_ATOMIC_SYMBOLS = {
    1: 'H',  2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
    6: 'C',  7: 'N',  8: 'O',  9: 'F',  10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S',  17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca',
    26: 'Fe', 29: 'Cu', 30: 'Zn', 35: 'Br', 53: 'I',
}


def _write_results_to_arc(results, output_dir, logger=None):
    """
    Write GeoDiff sampling results directly as arc files.

    Each molecule's conformations are written to a single arc file named after
    the molecule's SMILES string (special characters replaced with underscores).
    Mirrors the read_GeoDiff_result + write_arc logic from laspy GenFileParser.
    """
    os.makedirs(output_dir, exist_ok=True)

    for data in results:
        atom_type = data.atom_type
        atomic_numbers = atom_type.numpy() if torch.is_tensor(atom_type) else np.asarray(atom_type)
        num_atoms = len(atomic_numbers)

        smiles = getattr(data, 'smiles', None) or 'mol'
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', smiles)

        pos_gen = data.pos_gen
        if pos_gen is None:
            if logger:
                logger.warning('No pos_gen for %s, skipping arc output.' % smiles)
            continue

        if torch.is_tensor(pos_gen):
            pos_gen = pos_gen.numpy()
        else:
            pos_gen = np.asarray(pos_gen)

        if pos_gen.ndim == 3:
            # shape: (num_conf, num_atoms, 3)
            num_conf = pos_gen.shape[0]
            def _get_conf(pg, i, na): return pg[i]
        elif pos_gen.ndim == 2:
            # shape: (num_conf * num_atoms, 3)
            num_conf = pos_gen.shape[0] // num_atoms
            def _get_conf(pg, i, na): return pg[i * na : (i + 1) * na]
        else:
            if logger:
                logger.warning('Unexpected pos_gen shape %s for %s, skipping.' % (pos_gen.shape, smiles))
            continue

        arc_path = os.path.join(output_dir, safe_name + '.arc')
        with open(arc_path, 'w') as f:
            f.write('!BIOSYM archive 2\n')
            f.write('PBC=ON\n')

            for i in range(num_conf):
                conf_pos = _get_conf(pos_gen, i, num_atoms)   # (num_atoms, 3)

                mol_name = smiles if i == 0 else '%sconf%d' % (smiles, i + 1)

                f.write('     Energy  %8d    %8.2f     %12.6f\n' % (i, 0.0, 0.0))
                f.write('!DATE  %s\n' % mol_name)

                abc = np.max(conf_pos, axis=0) - np.min(conf_pos, axis=0) + 10.0
                f.write('PBC %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n' % (
                    abc[0], abc[1], abc[2], 90.0, 90.0, 90.0))

                for j, z in enumerate(atomic_numbers):
                    ele = _ATOMIC_SYMBOLS.get(int(z), 'X')
                    x, y, zc = conf_pos[j]
                    f.write('%s %15.9f %15.9f %15.9f CORE %4d %2s %2s %8.4f %4d\n' % (
                        ele.ljust(3), x, y, zc, j + 1, ele, ele, 0.0, j + 1))

                f.write('end\nend\n')


def sample(config):
    # ------------------------------------------------------------------ #
    # Unpack config                                                        #
    # ------------------------------------------------------------------ #
    device    = config.global_setting.device
    task_name = config.global_setting.task_name

    ckpt_path  = config.sample.ckpt
    start_idx  = getattr(config.sample, 'start_idx',  0)
    end_idx    = getattr(config.sample, 'end_idx',    int(1e9))
    n_steps    = config.sample.n_steps
    w_global   = config.sample.guidance_scale
    sigma_cut  = config.sample.sigma_cutoff
    save_traj  = getattr(config.sample, 'save_traj', False)
    batch_size = getattr(config.sample, 'conf_per_iter', 32)

    # Optional: path to a partial results pkl for resuming a sampling run
    resume_pkl = getattr(config.sample, 'resume_pkl', None)

    # num_confs: plain int → fixed count per molecule
    #            '2x'     → 2 × num reference conformations
    _nc = config.sample.num_confs
    if isinstance(_nc, str) and _nc.endswith('x'):
        num_confs_fn = lambda num_refs: num_refs * int(_nc[:-1])
    else:
        num_confs_fn = lambda num_refs: int(_nc)

    # ------------------------------------------------------------------ #
    # Checkpoint                                                           #
    # ------------------------------------------------------------------ #
    ckpt = torch.load(ckpt_path, map_location=device)
    log_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))

    seed_all(config.global_setting.seed)

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    output_dir = get_new_log_dir(log_dir, prefix='sample', tag=task_name)
    logger = get_logger('sample', output_dir)
    logger.info(config)

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    logger.info('Loading datasets...')
    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order),
    ])
    test_set = PackedConformationDataset(config.dataset.test, transform=transforms)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    logger.info('Loading model...')
    model = get_model(ckpt['config'].model).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # ------------------------------------------------------------------ #
    # Filter molecules by index range                                      #
    # ------------------------------------------------------------------ #
    test_set_selected = [
        data for i, data in enumerate(test_set)
        if start_idx <= i < end_idx
    ]

    # ------------------------------------------------------------------ #
    # Optionally resume from a partial results pkl                         #
    # ------------------------------------------------------------------ #
    done_smiles = set()
    results = []
    if resume_pkl is not None:
        with open(resume_pkl, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)

    # ------------------------------------------------------------------ #
    # Build expanded flat list: each molecule repeated num_confs times    #
    # ------------------------------------------------------------------ #
    flat_data    = []   # cloned data items (pos_ref cleared)
    flat_mol_idx = []   # which molecule index each item belongs to

    for i, data in enumerate(test_set_selected):
        if data.smiles in done_smiles:
            continue
        num_refs    = data.pos_ref.size(0) // data.num_nodes
        num_samples = num_confs_fn(num_refs)
        data_input  = data.clone()
        data_input['pos_ref'] = None
        for _ in range(num_samples):
            flat_data.append(data_input.clone())
            flat_mol_idx.append(i)

    logger.info('Total conformations to generate: %d (batch_size=%d)' % (len(flat_data), batch_size))

    # ------------------------------------------------------------------ #
    # Batch sampling loop                                                  #
    # ------------------------------------------------------------------ #
    # mol_idx -> list of per-conformation pos tensors (num_nodes, 3)
    mol_pos_gens  = defaultdict(list)
    # mol_idx -> list of per-conformation trajectory tensors (save_traj only)
    mol_traj_gens = defaultdict(list)

    clip_local = None

    for start in tqdm(range(0, len(flat_data), batch_size), desc='sampling batches'):
        chunk         = flat_data[start : start + batch_size]
        chunk_mol_idx = flat_mol_idx[start : start + batch_size]
        batch = Batch.from_data_list(chunk).to(device)

        for attempt in range(2):   # at most one retry with local clipping
            try:
                pos_init = torch.randn(batch.num_nodes, 3, device=device)
                pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                    atom_type=batch.atom_type,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False,          # already done in transforms
                    n_steps=n_steps,
                    step_lr=1e-6,
                    w_global=w_global,
                    global_start_sigma=sigma_cut,
                    clip_local=clip_local,
                )
                pos_gen    = pos_gen.cpu()
                batch_cpu  = batch.batch.cpu()

                # Split generated positions back to individual conformations
                for g, mol_idx in enumerate(chunk_mol_idx):
                    node_mask = batch_cpu == g
                    mol_pos_gens[mol_idx].append(pos_gen[node_mask])
                    if save_traj:
                        # pos_gen_traj: list of (total_nodes, 3) tensors, one per step
                        traj_g = torch.stack([step.cpu()[node_mask] for step in pos_gen_traj])
                        mol_traj_gens[mol_idx].append(traj_g)

                break   # success — exit retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning('FloatingPointError in batch, retrying with local clipping.')

    # ------------------------------------------------------------------ #
    # Assemble per-molecule results                                        #
    # ------------------------------------------------------------------ #
    for mol_idx in sorted(mol_pos_gens.keys()):
        data = test_set_selected[mol_idx]
        if save_traj:
            # trajs: list of (n_steps, num_nodes, 3)  →  (n_steps, num_nodes * num_confs, 3)
            data.pos_gen = torch.cat(mol_traj_gens[mol_idx], dim=1)
        else:
            data.pos_gen = torch.cat(mol_pos_gens[mol_idx], dim=0)
        results.append(data)
        done_smiles.add(data.smiles)

    # ------------------------------------------------------------------ #
    # Save all results in order                                            #
    # ------------------------------------------------------------------ #
    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving all samples to: %s' % save_path)

    def get_mol_key(data):
        for j, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return j
        return -1

    results.sort(key=get_mol_key)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    # ------------------------------------------------------------------ #
    # Write arc files                                                      #
    # ------------------------------------------------------------------ #
    arc_dir = os.path.join(output_dir, 'arc')
    logger.info('Writing arc files to: %s' % arc_dir)
    _write_results_to_arc(results, arc_dir, logger=logger)
    logger.info('Arc files written.')
