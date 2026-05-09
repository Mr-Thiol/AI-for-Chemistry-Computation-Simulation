import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_and_save_all_acsf(npy_file, save_filename='Ti_ACSF_All_Fingerprints.png'):
    # 1. 读取数据
    try:
        features = np.load(npy_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"找不到文件 {npy_file}，请检查路径。")
        return

    # 获取所有 Ti 原子的索引列表
    ti_indices = list(features.keys())
    num_ti = len(ti_indices)
    
    if num_ti == 0:
        print("数据中没有找到 Ti 原子。")
        return

    print(f"检测到 {num_ti} 个 Ti 原子，正在生成图表...")

    # 2. 动态创建画布 (高度根据原子数量自动调整)
    # 每个子图高度设为 3，总高度为 num_ti * 3
    fig, axes = plt.subplots(num_ti, 1, figsize=(12, 3 * num_ti), sharex=True)
    
    # 如果只有一个 Ti 原子，axes 不是数组，我们把它包装成列表以方便循环
    if num_ti == 1:
        axes = [axes]

    # 为了美观，给不同的柱子设定一组循环颜色
    colors = ['royalblue', 'darkorange', 'forestgreen', 'firebrick', 'purple', 'teal']

    # 3. 循环绘制每个 Ti 原子的条形码
    for i, idx in enumerate(ti_indices):
        fp = features[idx]
        ax = axes[i]
        color = colors[i % len(colors)] # 循环使用颜色
        
        ax.bar(range(len(fp)), fp, color=color, alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(f"Ti 原子 (原子索引: {idx}) 的 ACSF 结构指纹", fontsize=12)
        ax.set_ylabel("特征响应值", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # 只在最下面的一个子图显示 X 轴标签
        if i == num_ti - 1:
            ax.set_xlabel("特征维度索引 (Feature Dimension)", fontsize=12)

    # 调整子图之间的间距
    plt.tight_layout()

    # 4. 保存为高清图片 (DPI=300 是学术论文的标准分辨率)
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图表已成功保存为高清图片: {os.path.abspath(save_filename)}")

    # 5. 在屏幕上显示
    plt.show()

if __name__ == "__main__":
    # 请确保这里填入你上一步生成的 npy 文件名
    visualize_and_save_all_acsf('Ti_ACSF_features_Modular.npy')