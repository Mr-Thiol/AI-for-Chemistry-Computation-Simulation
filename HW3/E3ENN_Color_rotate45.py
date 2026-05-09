import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

# ==========================================
# 核心修复：解决 PyTorch 2.6 的安全拦截报错
# ==========================================
import torch
torch.serialization.add_safe_globals([slice])
from e3nn import o3

# 优化字体设置
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix' 

def extract_node_e3_features(atoms, center_idx, max_rc=7.0):
    i_indices, j_indices, d_ij, D_ij = neighbor_list('ijdD', atoms, cutoff=max_rc)
    
    mask = (i_indices == center_idx)
    vecs = D_ij[mask]
    dists = d_ij[mask]
    
    valid_mask = dists > 0
    vecs = vecs[valid_mask]
    dists = dists[valid_mask]
    
    r_hat = torch.tensor(vecs / dists[:, np.newaxis], dtype=torch.float32)
    
    sh_features = o3.spherical_harmonics(
        l=[0, 1, 2], 
        x=r_hat, 
        normalize=False, 
        normalization='integral'
    )
    
    node_feature = sh_features.sum(dim=0).numpy()
    return node_feature

def autolabel(ax, rects):
    """为柱状图自动添加数值标签"""
    for rect in rects:
        height = rect.get_height()
        display_val = 0.0 if abs(height) < 1e-4 else height
        y_pos = height + 0.05 if height >= 0 else height - 0.05
        va = 'bottom' if height >= 0 else 'top'
        
        ax.annotate(f'{display_val:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                    xytext=(0, 0),  
                    textcoords="offset points",
                    ha='center', va=va, fontsize=9, color='#333333',
                    fontweight='bold')

def main():
    print("正在加载并提取原始晶体的特征...")
    atoms_orig = read('TiO2_anatase.cif')
    
    print("正在构建沿 Z 轴旋转 90 度的晶体...")
    atoms_rot = atoms_orig.copy()
    
    # ==========================================
    # 真正的核心修复在这里：必须加上 rotate_cell=True
    # 保证晶胞和原子一起作为刚体旋转，防止 PBC 重新映射
    # ==========================================
    atoms_rot.rotate(45, 'z', center='COM', rotate_cell=True)
    
    ti_indices = [atom.index for atom in atoms_orig if atom.symbol == 'Ti']
    target_idx = ti_indices[0]
    
    feat_orig = extract_node_e3_features(atoms_orig, target_idx)
    feat_rot = extract_node_e3_features(atoms_rot, target_idx)
    
    print("\n========= 数值对比 =========")
    print(f"l=0 原始: {np.round(feat_orig[0:1], 4)} | 旋转: {np.round(feat_rot[0:1], 4)}")
    print(f"l=1 原始: {np.round(feat_orig[1:4], 4)} | 旋转: {np.round(feat_rot[1:4], 4)}")
    print(f"l=2 原始: {np.round(feat_orig[4:9], 4)} | 旋转: {np.round(feat_rot[4:9], 4)}")
    
    # 可视化重构部分
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 3, 5]})
    
    fig.suptitle(f"$E(3)$ Equivariance Validation: $\\mathrm{{TiO}}_2$ Ti-Atom (Index {target_idx}) rotated 90° via Z-axis", 
                 fontsize=16, fontweight='bold', y=0.98, fontfamily='Arial')
    
    labels = ['$l=0$ (Scalar)', '$l=1$ (Vector)', '$l=2$ (Tensor)']
    slices = [slice(0, 1), slice(1, 4), slice(4, 9)]
    
    color_orig = '#2A5C8A' 
    color_rot = '#E07A5F'
    
    y_max = max(np.max(feat_orig), np.max(feat_rot)) * 1.3
    y_min = min(np.min(feat_orig), np.min(feat_rot)) * 1.3
    if y_min > -0.5: y_min = -2.0 # 强制给负值留出空间，展示翻转

    for i in range(3):
        ax = axes[i]
        slc = slices[i]
        dim_len = slc.stop - slc.start
        
        x_labels = [f"Dim {j}" for j in range(dim_len)]
        x = np.arange(dim_len)
        width = 0.35
        
        rects1 = ax.bar(x - width/2, feat_orig[slc], width, 
                        label='Original', color=color_orig, edgecolor='none', zorder=3, alpha=0.9)
        rects2 = ax.bar(x + width/2, feat_rot[slc], width, 
                        label='Rotated 90° (Z)', color=color_rot, edgecolor='none', zorder=3, alpha=0.9)
        
        autolabel(ax, rects1)
        autolabel(ax, rects2)
        
        ax.set_title(labels[i], fontsize=13, pad=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11, fontstyle='italic')
        ax.set_ylim(y_min, y_max)
        
        ax.axhline(0, color='black', linewidth=1.2, zorder=2)
        ax.grid(axis='y', linestyle='-', alpha=0.3, color='#B0B0B0', zorder=1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')
        
        if i == 0:
            ax.set_ylabel("Spherical Harmonic Features", fontsize=12, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(1.5, -0.15), 
                      ncol=2, frameon=False, fontsize=11)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
            ax.spines['left'].set_visible(False)

    plt.subplots_adjust(wspace=0.1, bottom=0.2)
    
    save_path = 'E3_Equivariance_Proof_Optimized_Fixed_Rotate45.png'
    plt.savefig(save_path, dpi=400, bbox_inches='tight', transparent=False)
    print(f"\n✅ 高清可视化图表已生成: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()