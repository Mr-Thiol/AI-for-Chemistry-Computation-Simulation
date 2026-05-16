import copy
import torch
import pickle
import random
import argparse
import networkx as nx

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from torch_geometric.data import Data
from rdkit.Chem.rdchem import Bond, Mol
from rdkit.Chem.rdmolfiles import MolFromMolFile, MolToSmiles


def _bond_type_to_int(bond: Bond) -> int:
    # Keep same convention as common PyG molecular preprocessing.
    return int(bond.GetBondTypeAsDouble())


def _rdmol_to_data(mol: Mol, idx: int) -> Data:
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no conformer (no 3D coordinates found)")

    conf = mol.GetConformer(0)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    atom_type = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        t = _bond_type_to_int(bond)
        row.extend([i, j])
        col.extend([j, i])
        edge_type.extend([t, t])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    if edge_index.numel() > 0:
        n = mol.GetNumAtoms()
        perm = (edge_index[0] * n + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]

    smiles = MolToSmiles(mol)
    data = Data(
        edge_index=edge_index,
        pos=pos,
        atom_type=atom_type,
        edge_type=edge_type,
        rdmol=copy.deepcopy(mol),
        smiles=smiles,
        nx=nx.Graph(),
        idx=torch.tensor([idx], dtype=torch.long),
    )
    return data


def _read_arc_positions(arc_path: Path) -> List[List[float]]:
    """
    Read first structure coordinates from ARC file.
    Parsing follows the main idea of FileParser.read_arc:
    - A structure starts at lines containing "Energy" or "React"
    - Atom lines contain "CORE" or "XXXX"
    Returns empty list if no valid structure is found.
    """
    structures: List[List[List[float]]] = []
    current_coords: Optional[List[List[float]]] = None

    with arc_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ("Energy" in line) or ("React" in line):
                if current_coords is not None and current_coords:
                    structures.append(current_coords)
                current_coords = []
                continue

            if ("CORE" in line) or ("XXXX" in line):
                if current_coords is None:
                    continue
                parts = line.split()
                try:
                    if len(parts) >= 4:
                        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                    else:
                        xyz = [float(line[5:20]), float(line[20:35]), float(line[35:50])]
                except (ValueError, IndexError):
                    continue
                current_coords.append(xyz)

    if current_coords is not None and current_coords:
        structures.append(current_coords)
    return structures[0] if structures else []


def make_data(mol_dir: str, arc_dir: Optional[str] = None) -> List[Data]:
    mol_root = Path(mol_dir)
    if not mol_root.exists():
        raise FileNotFoundError(f"mol_dir does not exist: {mol_dir}")

    arc_root = Path(arc_dir) if arc_dir else None
    mol_files = sorted(mol_root.glob("*.mol"))
    if not mol_files:
        raise FileNotFoundError(f"no .mol files found under: {mol_dir}")

    dataset: List[Data] = []
    for i, mol_file in enumerate(tqdm(mol_files, desc="Building dataset")):
        mol = MolFromMolFile(str(mol_file), sanitize=True, removeHs=False)
        if mol is None:
            continue

        data = _rdmol_to_data(mol, idx=i)
        mol_atom_count = mol.GetNumAtoms()

        if arc_root is not None:
            arc_file = arc_root / f"{mol_file.stem}.arc"
            if arc_file.exists():
                arc_pos = _read_arc_positions(arc_file)
                if arc_pos and len(arc_pos) == mol_atom_count:
                    data.pos = torch.tensor(arc_pos, dtype=torch.float32)
                elif arc_pos:
                    print(
                        f"[warn] atom count mismatch for {mol_file.stem}: "
                        f"mol={mol_atom_count}, arc={len(arc_pos)}; keep mol coordinates."
                    )

        dataset.append(data)

    return dataset


def split_dataset(dataset: List[Data], divide: float, seed: int = 2021) -> tuple[List[Data], List[Data], List[Data]]:
    if not (0.0 < divide < 1.0):
        raise ValueError(f"divide must be in (0, 1), got {divide}")

    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * divide)
    train_data = shuffled[:cut]
    val_data = shuffled[cut:]
    test_data = shuffled
    return train_data, val_data, test_data


def save_datasets(train_data: List[Data], val_data: List[Data], test_data: List[Data], save_dir: str) -> dict[str, str]:
    result_files: dict[str, str] = {}
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]

    for dataset_type, data in datasets:
        if data:
            filename = f"{dataset_type}_{len(data)}.pkl"
            filepath = save_path / filename
            with filepath.open("wb") as f:
                pickle.dump(data, f)
            result_files[dataset_type] = str(filepath)
            print(f"Saved {filename} with {len(data)} samples")

    return result_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_dir", required=True, type=str, help="Directory containing .mol files")
    parser.add_argument("--arc_dir", default=None, type=str, help="Optional directory containing .arc files")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory for train/val/test pkl files")
    parser.add_argument("--divide", default=0.9, type=float, help="Train split ratio; val is the rest.")
    parser.add_argument("--seed", default=2026, type=int, help="Random seed for train/val split.")
    args = parser.parse_args()

    dataset = make_data(mol_dir=args.mol_dir, arc_dir=args.arc_dir)
    train_data, val_data, test_data = split_dataset(dataset, divide=args.divide, seed=args.seed)
    result_files = save_datasets(train_data, val_data, test_data, save_dir=args.out_dir)
    print(f"Done. Saved splits: {result_files}")

if __name__ == "__main__":
    main()
