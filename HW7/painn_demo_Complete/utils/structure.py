import os, sys
import copy

import math
import torch
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data

from utils.pbc import build_pbc, check_healthy

ELE_DICT = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
    'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24,
    'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
    'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88,
    'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104,
    'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
    'Uut': 113, 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'UUo': 118
}


def check_inside(pos, inverse_cell):
    # 使用逆矩阵计算点的笛卡尔坐标
    cartesian_point = np.dot(pos, inverse_cell)
    # 检查点是否在晶胞内
    is_inside = all(0 <= cartesian_point) and all(cartesian_point <= 1)
    return is_inside


def check_positions_in_cell(positions, cell):
    # Convert positions to fractional coordinates
    fractional_coords = torch.linalg.solve(cell.T, positions.T).T

    # Check if fractional coordinates are within [0, 1] for each position
    is_inside = torch.all((fractional_coords >= 0) & (fractional_coords <= 1), dim=1)

    return is_inside


def move_center_torch(pos, cell):
    fractional_coords = torch.linalg.solve(cell.T, pos.T).T
    adjusted_fractional_coords = fractional_coords % 1
    adjusted_positions = (cell.T @ adjusted_fractional_coords.T).T
    return adjusted_positions


def move_center_np(pos, cell):
    fractional_coords = np.linalg.solve(cell.T, pos)

    # Adjust fractional coordinates to be within 0 and 1
    adjusted_fractional_coords = fractional_coords % 1

    # Convert adjusted fractional coordinates back to Cartesian coordinates
    adjusted_position = cell.T.dot(adjusted_fractional_coords)

    return adjusted_position


def cheak_pos_in_pbc(idx, pos, cell):
    cell = np.array(cell)
    inverse_cell = np.linalg.inv(cell)
    shift_flag = False
    for id, _pos in enumerate(pos):
        _pos = np.array(_pos)
        is_inside = check_inside(_pos, inverse_cell)
        if is_inside:
            pass
        else:
            new_pos = move_center_np(_pos, cell)

            print("idx:", idx, "row", id, ":", _pos, "->", new_pos)
            pos[id] = new_pos
            if not check_inside(new_pos, inverse_cell): raise Exception("点不在晶胞内,逻辑异常")
            shift_flag = True
    return pos, shift_flag


def _min_zero(coor):
    if abs(coor) < 1e-8: return 0
    return coor


def Lat(line):
    pbc = [float(l) for l in line.split()[1:7]]
    a, b, c = pbc[0:3]
    alpha, beta, gamma = [x * np.pi / 180.0 for x in pbc[3:]]

    bc2 = b ** 2 + c ** 2 - 2 * b * c * math.cos(alpha)
    h1 = _min_zero(a)
    h2 = _min_zero(b * math.cos(gamma))
    h3 = _min_zero(b * math.sin(gamma))
    h4 = _min_zero(c * math.cos(beta))
    h5 = _min_zero(((h2 - h4) ** 2 + h3 ** 2 + c ** 2 - h4 ** 2 - bc2) / (2 * h3))
    h6 = _min_zero(math.sqrt(c ** 2 - h4 ** 2 - h5 ** 2))
    lat = [[h1, 0., 0.], [h2, h3, 0.], [h4, h5, h6]]
    return lat


def load_structure(raw_dir, force=False, cutoff=5.0):
    data_list = []
    idx = 0
    structure_file = os.path.join(raw_dir, "structure.arc")
    if force:
        energy_list, force_list, _force = [], [], []
        force_file = os.path.join(raw_dir, "force.arc")
        with open(force_file, "r") as f:
            for line in tqdm(f.readlines(), desc="force"):
                line = line.split()
                if line and line[0] == 'For':
                    energy_list.append(float(line[-1]))
                elif len(line) == 3:
                    fx, fy, fz = float(line[0]), float(line[1]), float(line[2])
                    _force.append((fx, fy, fz))
                elif len(line) == 6:
                    pass
                elif len(line) == 0:
                    if _force != []: force_list.append(_force)
                    _force = []

    with open(structure_file, "r") as f:
        structure_end = False
        for line in tqdm(f.readlines(), desc="structure"):
            if "CORE" in line:
                line = line.split()
                _z = float(ELE_DICT[line[0]])
                _pos = (float(line[1]), float(line[2]), float(line[3]))
                z.append(_z)
                pos.append(_pos)
            elif "Energy" in line:
                line = line.split()
                energy = float(line[3])
                structure_id = int(line[1])
            elif "PBC" in line and "ON" not in line:
                cell = Lat(line)
            elif "!DATE" in line:
                z, pos = [], []
            elif "end" in line:
                structure_end = not structure_end
            if structure_end:
                # 校验能量        =============================================================
                if energy != energy_list[idx]: raise Exception(f"[{idx}]能量不相等")
                if len(z) != len(force_list[idx]): raise Exception(f"[{idx}]数量不相等")
                # 校验位置 torch  =============================================================
                # float64 为了精确计算
                pos = torch.tensor(pos, dtype=torch.float64)
                cell = torch.tensor(cell, dtype=torch.float64)
                inside = check_positions_in_cell(pos, cell).all().item()
                if not inside:
                    pos = move_center_torch(pos, cell)
                    if not check_positions_in_cell(pos, cell).all().item(): raise Exception("逻辑异常")
                ############################################################################## 计算pbc位移
                pos = pos.to(torch.float)
                cell = cell.to(torch.float)
                shift, cell_num, i_num, j_num, pos_j = build_pbc(pos, cell, cutoff=cutoff)
                _force = torch.tensor(force_list[idx], dtype=torch.float) if force else None
                data = Data(
                    z=torch.tensor(z, dtype=torch.long),
                    pos=pos,
                    energy=torch.tensor(energy, dtype=torch.float),
                    force=_force,
                    cell=cell,
                    shift=shift,
                    cell_num=cell_num,
                    i_num=i_num,
                    j_num=j_num,
                    pos_j=pos_j,
                    idx=idx,
                    structure_id=structure_id,
                )
                check, _force, HEALTH = check_healthy(data, _force, cutoff)
                if _force is not None:
                    data.force = _force
                if check: data_list.append(data)
                idx += 1
    print(HEALTH)
    return data_list


if __name__ == '__main__':
    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    load_structure(os.path.join(FILE_PATH, "../data", "train", "raw"), force=True, cutoff=5.0)
