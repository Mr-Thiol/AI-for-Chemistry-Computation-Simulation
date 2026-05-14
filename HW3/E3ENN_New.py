import os
# ==========================================
# 修复 1：解决 Intel 和 PyTorch 的 OpenMP 底层冲突
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
# ==========================================
# 修复 2：解决 PyTorch 2.6 的安全拦截报错
# ==========================================
torch.serialization.add_safe_globals([slice])
from e3nn import o3

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

# 设置无衬线英文字体，适合学术图表
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在加载 TiO2 晶体并提取 E(3) 等变特征...")
    # 1. 读取晶体
    atoms = read('TiO2_anatase.cif')
    
    # 获取第一个 Ti 原子的索引
    ti_indices = [atom.index for atom in atoms if atom.symbol == 'Ti']
    target_idx = ti_indices[0]
    
    # 2. 计算邻居距离矢量
    max_rc = 7.0
    i_indices, j_indices, d_ij, D_ij = neighbor_list('ijdD', atoms, cutoff=max_rc)
    
    mask = (i_indices == target_idx)
    vecs = D_ij[mask]
    dists = d_ij[mask]
    
    # 排除距离为 0 的自身
    valid_mask = dists > 0
    vecs = vecs[valid_mask]
    dists = dists[valid_mask]
    
    # 计算方向单位向量 r_hat
    r_hat = torch.tensor(vecs / dists[:, np.newaxis], dtype=torch.float32)
    
    # 3. 使用 e3nn 计算球谐函数 (提取几何特征)
    print("正在通过 e3nn.o3 计算 l=0, 1, 2 的球谐函数特征...")
    sh_features = o3.spherical_harmonics(
        l=[0, 1, 2], 
        x=r_hat, 
        normalize=False, 
        normalization='integral'
    )
    
    # 4. Message Passing: 聚合邻居特征 (Sum Pooling)
    node_feature = sh_features.sum(dim=0).numpy()
    
    print("\n====== Ti 原子提取完毕 ======")
    print(f"总特征维度: {len(node_feature)} 维 (1 + 3 + 5)")
    print(f"l=0 (标量): {np.round(node_feature[0:1], 4)}")
    print(f"l=1 (向量): {np.round(node_feature[1:4], 4)}  <-- 注意看，因为高度对称性几乎为0！")
    print(f"l=2 (张量): {np.round(node_feature[4:9], 4)}")

    # ==========================================
    # 5. 高清学术可视化 (已优化布局与顶头问题)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 将特征分为三个区域
    dims = np.arange(9)
    
    # 绘制背景色块以区分不同阶数的特征
    ax.axvspan(-0.5, 0.5, facecolor='#e8f4f8', alpha=0.8, zorder=0)
    ax.axvspan(0.5, 3.5, facecolor='#e8fae8', alpha=0.8, zorder=0)
    ax.axvspan(3.5, 8.5, facecolor='#f3e8fa', alpha=0.8, zorder=0)
    
    # 绘制柱状图
    ax.bar(dims, node_feature, color='#2c3e50', width=0.6, zorder=2)
    
    # 启用 SymLog (对数/线性对称轴) 
    ax.set_yscale('symlog', linthresh=1e-4)

    # 【核心修复】：动态计算并显式设置 Y 轴的上下限，给文字和最高柱子留出充足空间
    max_abs_val = np.max(np.abs(node_feature))
    
    # 在对数坐标下，* 100 意味着向上多留出两个数量级的留白空间
    ymin = -max_abs_val * 2
    ymax = max_abs_val * 200 
    ax.set_ylim(ymin, ymax)
    
    # 添加区域顶部文字 (将 va 改为 'bottom' 且位置适中，避免与柱子重叠)
    y_text_pos = max_abs_val * 5
    ax.text(0, y_text_pos, "l=0 (Scalar)\n1 dim", ha='center', va='bottom', fontweight='bold', color='#4a4a4a')
    ax.text(2, y_text_pos, "l=1 (Vector)\n3 dims", ha='center', va='bottom', fontweight='bold', color='#4a4a4a')
    ax.text(6, y_text_pos, "l=2 (Tensor)\n5 dims", ha='center', va='bottom', fontweight='bold', color='#4a4a4a')

    # 添加白色分割线
    ax.axvline(x=0.5, color='white', linewidth=1.5, zorder=1)
    ax.axvline(x=3.5, color='white', linewidth=1.5, zorder=1)

    # 设置美观的标签和标题
    ax.set_xlim(-0.5, 8.5)
    
    # X 轴自定义刻度标签，展现物理意义
    x_labels = ['m=0', 'm=-1', 'm=0', 'm=1', 'm=-2', 'm=-1', 'm=0', 'm=1', 'm=2']
    ax.set_xticks(dims)
    ax.set_xticklabels(x_labels)
    
    ax.set_title("E(3)-Equivariant Features (Spherical Harmonics) for Ti Atom", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Feature Response (Symlog Scale)", fontsize=12)
    ax.set_xlabel("Spherical Harmonic Degree (m)", fontsize=12)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    
    # 调整并保存
    save_filename = 'Ti_E3_Features_Optimized.png'
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化图表已优化并保存至: {os.path.abspath(save_filename)}")
    plt.show()

if __name__ == "__main__":
    main()