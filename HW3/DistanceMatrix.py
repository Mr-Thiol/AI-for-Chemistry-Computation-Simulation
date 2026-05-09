# script_1_distance_matrix_dual_output.py
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
import pickle
import json

def calculate_and_save_distances(cif_path, max_rc):
    """
    读取 CIF 文件，计算考虑 PBC 的距离，并同时保存为 pkl 和 json 格式。
    """
    print(f"正在读取结构文件: {cif_path} ...")
    atoms = read(cif_path)
    symbols = np.array(atoms.get_chemical_symbols())
    
    print(f"正在计算截断半径 {max_rc} Å 内的邻居信息 (已自动处理 PBC) ...")
    i_indices, j_indices, d_ij, D_ij = neighbor_list('ijdD', atoms, cutoff=max_rc)
    
    data_for_pkl = {}
    data_for_json = {}
    
    for idx in range(len(atoms)):
        mask = (i_indices == idx)
        
        # ==========================================
        # 数据集 A：面向高性能计算 (保留 NumPy 数组)
        # ==========================================
        data_for_pkl[idx] = {
            'symbol': symbols[idx],
            'neighbor_indices': j_indices[mask],
            'neighbor_symbols': symbols[j_indices[mask]],
            'distances': d_ij[mask],
            'distance_vectors': D_ij[mask]
        }
        
        # ==========================================
        # 数据集 B：面向人类阅读与作业提交 (列表 + 截断小数)
        # ==========================================
        data_for_json[int(idx)] = {
            'symbol': str(symbols[idx]),
            'neighbor_indices': j_indices[mask].tolist(),
            'neighbor_symbols': symbols[j_indices[mask]].tolist(),
            # 保留4位小数，让 JSON 文件更加清爽整洁
            'distances': [round(float(d), 4) for d in d_ij[mask]],
            'distance_vectors': [[round(float(v), 4) for v in vec] for vec in D_ij[mask]]
        }
    
    # 保存高效计算文件
    pkl_file = 'neighbor_data.pkl'
    print(f"正在保存高性能计算格式至: {pkl_file} ...")
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_for_pkl, f)
        
    # 保存人类可读文件
    json_file = 'neighbor_data_readable.json'
    print(f"正在保存人类可读/作业提交格式至: {json_file} ...")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data_for_json, f, indent=4, ensure_ascii=False)
        
    print("\n🎉 全部保存成功！现在你可以用文本编辑器查看 json 文件，同时继续用 pkl 跑脚本 2 了。")

if __name__ == "__main__":
    calculate_and_save_distances(cif_path='TiO2_anatase.cif', max_rc=7.0)