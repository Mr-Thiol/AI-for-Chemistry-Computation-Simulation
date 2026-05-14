import numpy as np
import matplotlib.pyplot as plt
import os

# 使用无衬线学术英文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def visualize_optimized_acsf(npy_file, feature_lengths, save_filename='Ti_ACSF_Final.png', use_symlog=True):
    # 1. 加载数据
    try:
        features = np.load(npy_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"File not found: {npy_file}")
        return

    ti_indices = list(features.keys())
    num_ti = len(ti_indices)
    if num_ti == 0: return

    # 2. 计算各分区索引与总维度
    partitions = {}
    current_idx = 0
    total_dims = sum(feature_lengths.values()) # 严格计算总维度
    
    for name, length in feature_lengths.items():
        if length > 0:
            partitions[name] = (current_idx, current_idx + length)
            current_idx += length

    # 3. 创建画布
    fig, axes = plt.subplots(num_ti, 1, figsize=(14, 3.5 * num_ti), sharex=True)
    if num_ti == 1: axes = [axes]

    bg_colors = {
        'S1': '#e8f4f8', 'S2': '#f3e8fa', 'S3': '#fdf0d5',
        'S4': '#e8fae8', 'S5': '#ffe6e6', 'S6': '#e6ecef'
    }
    bar_color = '#2c3e50'

    # 4. 绘图循环
    for i, idx in enumerate(ti_indices):
        fp = features[idx]
        ax = axes[i]
        
        # 画柱状图
        ax.bar(range(len(fp)), fp, color=bar_color, alpha=0.85, width=0.6, edgecolor='none', zorder=2)
        
        # ==========================================
        # 核心优化 3：对称对数坐标轴 (SymLog)
        # ==========================================
        if use_symlog:
            # linthresh=1e-2 意味着 -0.01 到 0.01 之间是线性坐标，之外是对数坐标
            ax.set_yscale('symlog', linthresh=1e-2)
            # 调整上限，留出写文字的空间
            y_max = ax.get_ylim()[1]
            ax.set_ylim(-1e-3, y_max * 10) # 对数轴的顶部空间需要用乘法放大
            ax.set_ylabel("Response (SymLog Scale)", fontsize=11, fontweight='medium')
        else:
            y_max = ax.get_ylim()[1]
            ax.set_ylim(0, y_max * 1.25)
            ax.set_ylabel("Feature Response", fontsize=11, fontweight='medium')
            
        new_y_max = ax.get_ylim()[1]

        for name, (start, end) in partitions.items():
            # 背景色块
            ax.axvspan(start - 0.5, end - 0.5, facecolor=bg_colors.get(name, '#eeeeee'), alpha=0.8, zorder=0)
            
            # ==========================================
            # 核心优化 2：缩小白边缝隙
            # ==========================================
            if end < total_dims:
                ax.axvline(x=end - 0.5, color='#ffffff', linewidth=0.6, zorder=1)
            
            if i == 0:
                mid_point = (start + end - 1) / 2
                # 根据是否为对数轴调整文字高度位置
                text_y = new_y_max * 0.5 if use_symlog else new_y_max * 0.95
                ax.text(mid_point, text_y, f"{name}\n({end-start} dims)", 
                        ha='center', va='top', fontsize=10, fontweight='bold', color='#4a4a4a')

        # ==========================================
        # 核心优化 1：消除右侧神秘空白
        # ==========================================
        ax.set_xlim(-0.5, total_dims - 0.5)
        
        ax.set_title(f"ACSF Structural Fingerprint for Ti Atom (Index: {idx})", fontsize=13, fontweight='bold', pad=15)
        
        # 优化网格线
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#bdc3c7', zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        if i == num_ti - 1:
            ax.set_xlabel("Feature Dimension Index", fontsize=12, fontweight='medium')

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Optimized chart saved to: {os.path.abspath(save_filename)}")
    plt.show()

if __name__ == "__main__":
    my_feature_lengths = {
        'S1': 47,
        'S2': 30,
        'S3': 24,
        'S4': 48,
        'S5': 36,
        'S6': 16
    }
    
    # 启用 use_symlog=True 解决大小跨度问题
    visualize_optimized_acsf('Ti_ACSF_Full_Features_S1toS6.npy', my_feature_lengths, use_symlog=True)