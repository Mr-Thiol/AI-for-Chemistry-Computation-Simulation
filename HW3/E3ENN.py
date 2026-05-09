import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

# ==========================================
# 核心修复：解决 PyTorch 2.6 的安全拦截报错
# 必须放在 import e3nn 之前！
# ==========================================
import torch
torch.serialization.add_safe_globals([slice])
from e3nn import o3

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_node_e3_features(atoms, center_idx, max_rc=7.0):
    """
    给定 ASE atoms 对象和原子索引，自动计算其周边的球谐函数等变特征
    """
    # 1. 寻找近邻并获取距离矢量
    i_indices, j_indices, d_ij, D_ij = neighbor_list('ijdD', atoms, cutoff=max_rc)
    
    mask = (i_indices == center_idx)
    vecs = D_ij[mask]
    dists = d_ij[mask]
    
    # 排除距离为 0 的自身
    valid_mask = dists > 0
    vecs = vecs[valid_mask]
    dists = dists[valid_mask]
    
    # 2. 计算方向单位向量
    # 注意：为了让网络具有平移不变性，只能用相对向量 vecs
    r_hat = torch.tensor(vecs / dists[:, np.newaxis], dtype=torch.float32)
    
    # 3. 使用 e3nn 计算球谐函数
    # l=0 (标量，1维), l=1 (向量，3维), l=2 (二阶张量，5维)
    sh_features = o3.spherical_harmonics(
        l=[0, 1, 2], 
        x=r_hat, 
        normalize=False, 
        normalization='integral'
    )
    
    # 4. Message Passing: 聚合邻居特征给中心原子
    node_feature = sh_features.sum(dim=0).numpy()
    return node_feature

def main():
    # 1. 读取原始晶体
    print("正在加载并提取原始晶体的特征...")
    atoms_orig = read('TiO2_anatase.cif')
    
    # 2. 构造旋转后的晶体
    print("正在构建沿 Z 轴旋转 90 度的晶体...")
    atoms_rot = atoms_orig.copy()
    # 沿 z 轴旋转 90 度，中心点设为原点或质心均可
    atoms_rot.rotate(90, 'z', center='COM')
    
    # 选择第一个 Ti 原子
    ti_indices = [atom.index for atom in atoms_orig if atom.symbol == 'Ti']
    target_idx = ti_indices[0]
    
    # 3. 分别提取特征
    feat_orig = extract_node_e3_features(atoms_orig, target_idx)
    feat_rot = extract_node_e3_features(atoms_rot, target_idx)
    
    # 打印数值对比
    print("\n========= 数值对比 =========")
    print("【l=0 标量部分】(期望: 完全不变)")
    print(f"原始: {np.round(feat_orig[0:1], 4)}")
    print(f"旋转: {np.round(feat_rot[0:1], 4)}")
    
    print("\n【l=1 向量部分】(期望: 随坐标系旋转，X/Y分量互换并可能变号，Z不变)")
    print(f"原始: {np.round(feat_orig[1:4], 4)}")
    print(f"旋转: {np.round(feat_rot[1:4], 4)}")
    
    # 4. 可视化对比
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f"Ti 原子 (索引 {target_idx}) 的 E(3) 等变特征：旋转 90° 对比", fontsize=14, fontweight='bold')
    
    labels = ['l=0 (标量)', 'l=1 (向量)', 'l=2 (二阶张量)']
    slices = [slice(0, 1), slice(1, 4), slice(4, 9)]
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    
    for i in range(3):
        ax = axes[i]
        slc = slices[i]
        x_labels = [f"Dim {j}" for j in range(slc.start, slc.stop)]
        x = np.arange(len(x_labels))
        width = 0.35
        
        ax.bar(x - width/2, feat_orig[slc], width, label='原始构型', color=colors[i], alpha=0.8)
        ax.bar(x + width/2, feat_rot[slc], width, label='Z轴旋转 90°', color='grey', alpha=0.6, hatch='//')
        
        ax.set_title(labels[i])
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig('E3_Equivariance_Proof.png', dpi=300)
    print("\n✅ 可视化图表已生成: E3_Equivariance_Proof.png")
    plt.show()

if __name__ == "__main__":
    main()