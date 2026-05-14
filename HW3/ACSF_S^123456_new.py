import numpy as np
import pandas as pd
import pickle
import glob
import os
import scipy.special as sp

# ==========================================
# 1. 基础数学与几何函数
# ==========================================
def fc(r, rc):
    """平滑截断函数 (Eq S4)"""
    return np.where(r <= rc, 0.5 * (np.tanh(1.0 - r / rc))**3, 0.0)

def R_n(r, rc, n):
    """径向基础函数 (Eq S5)"""
    return (r ** n) * fc(r, rc)

def calc_cos_theta(v_j, v_k, d_j, d_k):
    """计算两个邻居向量之间的夹角余弦值"""
    cos_theta = np.dot(v_j, v_k) / (d_j * d_k)
    return np.clip(cos_theta, -1.0, 1.0)

def calc_cos_delta(v_j, v_k, v_l):
    """计算四体相互作用的二面角余弦值 (cos delta_ijkl)"""
    n1 = np.cross(v_j, v_k)
    n2 = np.cross(v_k, v_l)
    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 1.0
    cos_delta = np.dot(n1, n2) / (norm1 * norm2)
    return np.clip(cos_delta, -1.0, 1.0)

def get_col(row, *possible_names):
    """辅助函数：处理表头中可能存在的希腊字母或英文别名"""
    for name in possible_names:
        if name in row:
            return row[name]
    raise KeyError(f"在表头中找不到列: {possible_names}")

# ==========================================
# 2. 独立特征计算模块 (S1 ~ S6)
# ==========================================
def calculate_S1(nbr_info, df):
    """计算 S^1"""
    features = []
    dists, syms = nbr_info['distances'], nbr_info['neighbor_symbols']
    
    for _, row in df.iterrows():
        rc, nb, n = float(row['rc']), str(row['nb']).strip(), float(row['n'])
        valid_mask = np.ones(len(syms), dtype=bool) if nb == 'All' else (syms == nb)
        valid_dists = dists[valid_mask]
        val = np.sum(R_n(valid_dists, rc, n)) if len(valid_dists) > 0 else 0.0
        features.append(val)
    return features

def calculate_S2(nbr_info, df):
    """计算 S^2 (包含 SciPy 版本兼容)"""
    features = []
    dists, syms, vecs = nbr_info['distances'], nbr_info['neighbor_symbols'], nbr_info['distance_vectors']
    
    for _, row in df.iterrows():
        rc, nb = float(row['rc']), str(row['nb']).strip()
        n, L = float(row['n']), int(row['L'])
        
        idx = np.where((syms == nb) | (nb == 'All'))[0]
        val = 0.0
        
        for m in range(-L, L + 1):
            sum_Y = 0j 
            for j in idx:
                if dists[j] > rc: continue
                r = dists[j]
                x, y, z = vecs[j]
                phi = np.arccos(np.clip(z / r, -1.0, 1.0))
                theta = np.arctan2(y, x)
                
                # 兼容新老 SciPy 版本
                if hasattr(sp, 'sph_harm_y'):
                    Y_Lm = sp.sph_harm_y(L, m, phi, theta) 
                else:
                    Y_Lm = sp.sph_harm(m, L, theta, phi)
                    
                sum_Y += R_n(r, rc, n) * Y_Lm
            val += np.abs(sum_Y)**2
        features.append(val)
    return features

def calculate_S3(nbr_info, df):
    """计算 S^3"""
    features = []
    dists, syms, vecs = nbr_info['distances'], nbr_info['neighbor_symbols'], nbr_info['distance_vectors']
    
    for _, row in df.iterrows():
        rc = float(row['rc'])
        nb1, nb2 = str(row['nb1']).strip(), str(row['nb2']).strip()
        n, m_p = float(row['n']), float(row['m'])
        zeta = float(get_col(row, 'ζ', 'zeta'))
        lam = float(get_col(row, 'λ', 'lam', 'lambda'))
        
        val, idx1, idx2 = 0.0, np.where((syms == nb1) | (nb1 == 'All'))[0], np.where((syms == nb2) | (nb2 == 'All'))[0]
        
        for j in idx1:
            if dists[j] > rc: continue
            for k in idx2:
                if j == k or dists[k] > rc: continue
                cos_theta = calc_cos_theta(vecs[j], vecs[k], dists[j], dists[k])
                term = (1.0 + lam * cos_theta)**zeta * R_n(dists[j], rc, n) * R_n(dists[k], rc, m_p)
                val += term
        features.append((2**(1.0 - zeta)) * val)
    return features

def calculate_S4(nbr_info, df):
    """计算 S^4"""
    features = []
    dists, syms, vecs = nbr_info['distances'], nbr_info['neighbor_symbols'], nbr_info['distance_vectors']
    
    for _, row in df.iterrows():
        rc = float(row['rc'])
        nb1, nb2 = str(row['nb1']).strip(), str(row['nb2']).strip()
        n, m_p, p = float(row['n']), float(row['m']), float(row['p'])
        zeta = float(get_col(row, 'ζ', 'zeta'))
        lam = float(get_col(row, 'λ', 'lam', 'lambda'))
        
        val, idx1, idx2 = 0.0, np.where((syms == nb1) | (nb1 == 'All'))[0], np.where((syms == nb2) | (nb2 == 'All'))[0]
        
        for j in idx1:
            if dists[j] > rc: continue
            for k in idx2:
                if j == k or dists[k] > rc: continue
                r_jk = np.linalg.norm(vecs[k] - vecs[j])
                cos_theta = calc_cos_theta(vecs[j], vecs[k], dists[j], dists[k])
                term = (1.0 + lam * cos_theta)**zeta * R_n(dists[j], rc, n) * R_n(dists[k], rc, m_p) * R_n(r_jk, rc, p)
                val += term
        features.append((2**(1.0 - zeta)) * val)
    return features

def calculate_S5(nbr_info, df):
    """计算 S^5"""
    features = []
    dists, syms, vecs = nbr_info['distances'], nbr_info['neighbor_symbols'], nbr_info['distance_vectors']
    
    for _, row in df.iterrows():
        rc = float(row['rc'])
        nb1, nb2 = str(row['nb1']).strip(), str(row['nb2']).strip()
        L, n, m_p, p = int(row['L']), float(row['n']), float(row['m']), float(row['p'])
        
        val, idx1, idx2 = 0.0, np.where((syms == nb1) | (nb1 == 'All'))[0], np.where((syms == nb2) | (nb2 == 'All'))[0]
        
        for j in idx1:
            if dists[j] > rc: continue
            for k in idx2:
                if j == k or dists[k] > rc: continue
                r_jk = np.linalg.norm(vecs[k] - vecs[j])
                cos_theta = calc_cos_theta(vecs[j], vecs[k], dists[j], dists[k])
                P_L = sp.eval_legendre(L, cos_theta)
                term = R_n(dists[j], rc, n) * R_n(dists[k], rc, m_p) * R_n(r_jk, rc, p) * P_L
                val += term
        features.append(val)
    return features

def calculate_S6(nbr_info, df):
    """计算 S^6"""
    features = []
    dists, syms, vecs = nbr_info['distances'], nbr_info['neighbor_symbols'], nbr_info['distance_vectors']
    
    for _, row in df.iterrows():
        rc = float(row['rc'])
        nb1, nb2, nb3 = str(row['nb1']).strip(), str(row['nb2']).strip(), str(row['nb3']).strip()
        n, m_p, p = float(row['n']), float(row['m']), float(row['p'])
        zeta = float(get_col(row, 'ζ', 'zeta'))
        lam = float(get_col(row, 'λ', 'lam', 'lambda'))
        
        val = 0.0
        idx1 = np.where((syms == nb1) | (nb1 == 'All'))[0]
        idx2 = np.where((syms == nb2) | (nb2 == 'All'))[0]
        idx3 = np.where((syms == nb3) | (nb3 == 'All'))[0]
        
        for j in idx1:
            if dists[j] > rc: continue
            for k in idx2:
                if j == k or dists[k] > rc: continue
                for l in idx3:
                    if l == j or l == k or dists[l] > rc: continue
                    cos_delta = calc_cos_delta(vecs[j], vecs[k], vecs[l])
                    term = (1.0 + lam * cos_delta)**zeta * R_n(dists[j], rc, n) * R_n(dists[k], rc, m_p) * R_n(dists[l], rc, p)
                    val += term
        features.append((2**(1.0 - zeta)) * val)
    return features

# ==========================================
# 3. 文件自动寻址与主控流
# ==========================================
def load_csv_for_Sn(n):
    pattern = f"Table_S*_S^{n}.csv"
    matched_files = glob.glob(pattern)
    if not matched_files:
        print(f"未找到匹配 {pattern} 的文件，将跳过 S^{n} 的计算。")
        return None
    
    target_file = matched_files[0]
    print(f"已加载参数表: {target_file}")
    df = pd.read_csv(target_file)
    df.columns = df.columns.str.strip()
    return df

def main():
    try:
        with open('neighbor_data.pkl', 'rb') as f:
            neighbor_data = pickle.load(f)
    except FileNotFoundError:
        print("错误: 未找到 neighbor_data.pkl，请先运行 脚本1。")
        return

    # 1. 加载参数表
    dfs = {
        'S1': load_csv_for_Sn('1'),
        'S2': load_csv_for_Sn('2'),
        'S3': load_csv_for_Sn('3'),
        'S4': load_csv_for_Sn('4'),
        'S5': load_csv_for_Sn('5'),
        'S6': load_csv_for_Sn('6')
    }
    
    # 2. 统计维度
    feature_lengths = {}
    for name, df in dfs.items():
        feature_lengths[name] = len(df) if df is not None else 0

    ti_indices = [idx for idx, info in neighbor_data.items() if info['symbol'] == 'Ti']
    print(f"\n开始提取特征，检测到的维度分布为: {feature_lengths}")
    
    all_features = {}
    for idx in ti_indices:
        info = neighbor_data[idx]
        current_feature = []
        
        if dfs['S1'] is not None: current_feature.extend(calculate_S1(info, dfs['S1']))
        if dfs['S2'] is not None: current_feature.extend(calculate_S2(info, dfs['S2']))
        if dfs['S3'] is not None: current_feature.extend(calculate_S3(info, dfs['S3']))
        if dfs['S4'] is not None: current_feature.extend(calculate_S4(info, dfs['S4']))
        if dfs['S5'] is not None: current_feature.extend(calculate_S5(info, dfs['S5']))
        if dfs['S6'] is not None: current_feature.extend(calculate_S6(info, dfs['S6']))
            
        all_features[idx] = np.array(current_feature)
        print(f"Ti 原子 {idx} 计算完成，当前全量特征总维度: {len(current_feature)}")
    
    # 3. 打包并保存
    export_data = {
        'features': all_features,
        'lengths': feature_lengths
    }
    save_filename = 'Ti_ACSF_Smart_Features.npy'
    np.save(save_filename, export_data)
    print(f"\n✅ 所有特征计算完毕，已附带维度字典保存至 '{save_filename}'！")

if __name__ == "__main__":
    main()