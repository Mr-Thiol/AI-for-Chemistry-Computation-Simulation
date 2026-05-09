import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_acsf_with_partitions(npy_file, feature_lengths, save_filename='Ti_ACSF_Partitioned.png'):
    # 1. 读取数据
    try:
        features = np.load(npy_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"找不到文件 {npy_file}，请检查路径。")
        return

    ti_indices = list(features.keys())
    num_ti = len(ti_indices)
    if num_ti == 0: return

    # 2. 计算每个分区在 X 轴上的起始和结束位置
    # 例如：S1(0~13), S3(13~37), S4(37~85), S6(85~101)
    partitions = {}
    current_idx = 0
    for name, length in feature_lengths.items():
        if length > 0:
            partitions[name] = (current_idx, current_idx + length)
            current_idx += length

    # 3. 创建画布
    fig, axes = plt.subplots(num_ti, 1, figsize=(12, 3 * num_ti), sharex=True)
    if num_ti == 1: axes = [axes]

    # 定义每个分区的背景颜色 (使用柔和的浅色)
    bg_colors = {'S1': '#e6f2ff', 'S3': '#fff0e6', 'S4': '#e6ffe6', 'S6': '#f2e6ff'}
    bar_color = '#2c3e50' # 统一的深色柱子显得更专业

    # 4. 绘图循环
    for i, idx in enumerate(ti_indices):
        fp = features[idx]
        ax = axes[i]
        
        # 画柱状图
        ax.bar(range(len(fp)), fp, color=bar_color, alpha=0.8, width=0.6)
        
        # 添加背景色块和顶部文字标注
        y_max = ax.get_ylim()[1]
        for name, (start, end) in partitions.items():
            # 画背景色块
            ax.axvspan(start - 0.5, end - 0.5, facecolor=bg_colors.get(name, '#eeeeee'), alpha=0.6, zorder=0)
            
            # 在色块顶部居中写上 S1, S3 等文字
            mid_point = (start + end - 1) / 2
            # 仅在第一张图（顶部）添加分区标签，避免重复显得杂乱
            if i == 0:
                ax.text(mid_point, y_max * 1.05, f"{name} 区\n({end-start}维)", 
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

        ax.set_title(f"Ti 原子 (索引: {idx}) 的 ACSF 结构指纹", fontsize=12, pad=25)
        ax.set_ylabel("特征响应值", fontsize=10)
        ax.set_xlim(-1, len(fp))
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        if i == num_ti - 1:
            ax.set_xlabel("特征维度索引 (Feature Dimension Index)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 带分区标注的图表已保存至: {os.path.abspath(save_filename)}")
    plt.show()

if __name__ == "__main__":
    # ⚠️⚠️⚠️ 重要：请将下面的数字改成你实际 CSV 文件的参数行数 (不含表头)！
    # 如果某个表格没算，或者找不到文件，把它写成 0 即可。
    my_feature_lengths = {
        'S1': 47,   # Table S3 的有效行数
        'S3': 24,   # Table S5 的有效行数
        'S4': 48,   # Table S6 的有效行数
        'S6': 16    # Table S8 的有效行数
    }
    
    visualize_acsf_with_partitions('Ti_ACSF_features_Modular.npy', my_feature_lengths)