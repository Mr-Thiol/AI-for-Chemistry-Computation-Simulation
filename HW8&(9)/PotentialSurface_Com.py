import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，确保图表正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 构造坐标网格
# ==========================================
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)

# ==========================================
# 2. 构造具有3个不等极小值点的复杂势能面
# ==========================================
# 基础势能：一个平缓的二维谐振子势，保证体系边界能量升高，形成宏观的势盆
E_base = 0.1 * (X**2 + Y**2)

# 极小值 1：全局极小值，深度最大 (最稳定的产物/反应物)
E_min1 = -5.0 * np.exp(-((X - 0)**2 + (Y - 2)**2) / (2 * 0.8**2))

# 极小值 2：局部极小值 A，深度中等 (亚稳态中间体)
E_min2 = -3.0 * np.exp(-((X - 2)**2 + (Y + 1)**2) / (2 * 0.6**2))

# 极小值 3：局部极小值 B，深度最小 (能量较高的亚稳态)
E_min3 = -2.0 * np.exp(-((X + 2)**2 + (Y + 1)**2) / (2 * 0.7**2))

# 总势能面叠加
Z = E_base + E_min1 + E_min2 + E_min3

# ==========================================
# 3. 绘制 3D 图
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用 coolwarm 配色，蓝色代表能量低(坑)，红色代表能量高
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.9)

ax.set_title('复杂势能面 (包含3个不等极小值点)', fontsize=15, pad=20)
ax.set_xlabel('坐标 X', fontsize=12)
ax.set_ylabel('坐标 Y', fontsize=12)
ax.set_zlabel('能量 E', fontsize=12)

# 调整视角，elev控制仰角(俯视)，azim控制方位角(旋转)
# 这个角度可以清楚地看到一个大坑和两个小坑
ax.view_init(elev=45, azim=120)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

# ==========================================
# 4. 保存图片并显示
# ==========================================
plt.tight_layout()
output_filename = 'complex_pes_3minima.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"绘制完成！图片已保存为 {output_filename}")

# 如果您在交互式环境中运行，取消下面这行的注释可以直接预览
plt.show()