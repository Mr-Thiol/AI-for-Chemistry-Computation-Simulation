import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 设置中文字体，确保图表正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 定义坐标网格 (x, y)
# ==========================================
# 谐振子和三角函数网格
x_lin = np.linspace(-3, 3, 50)
y_lin = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x_lin, y_lin)

# Lennard-Jones 和 机器学习专属网格 (避开 r 接近 0 的奇点)
x_lj = np.linspace(0.9, 3.0, 40)
y_lj = np.linspace(0.9, 3.0, 40)
X_lj, Y_lj = np.meshgrid(x_lj, y_lj)

# ==========================================
# 2. 计算四种势能面 Z
# ==========================================

# (1) 谐振子势 (Harmonic Oscillator)
k = 1.0
Z_harm = 0.5 * k * (X**2 + Y**2)

# (2) 三角函数势 (Trig Potential) 
Z_trig = -np.cos(X) - np.cos(Y)

# (3) Lennard-Jones 势
epsilon = 1.0
sigma = 1.0
def calc_lj(r):
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

Z_lj = calc_lj(X_lj) + calc_lj(Y_lj)
# 限制一下 Z 轴的最大值，防止斥力部分数值过大导致图像失真
Z_lj = np.clip(Z_lj, -2, 5) 

# (4) 机器学习势 (MLP) 
X_train = np.c_[X_lj.ravel(), Y_lj.ravel()]
y_train = Z_lj.ravel()

print("正在训练机器学习势函数，请稍候...")
mlp = MLPRegressor(hidden_layer_sizes=(30, 30), activation='tanh', solver='adam', 
                   max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# 预测新的势能面
Z_ml = mlp.predict(X_train).reshape(X_lj.shape)


# ==========================================
# 3. 绘制 3D 图像并调整视角
# ==========================================
fig = plt.figure(figsize=(16, 12))

# 绘图通用配置函数
def plot_3d_surface(ax, X, Y, Z, title, z_min, z_max):
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('坐标 X', fontsize=12)
    ax.set_ylabel('坐标 Y', fontsize=12)
    ax.set_zlabel('能量 E', fontsize=12)
    ax.set_zlim(z_min, z_max)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

# (1) 谐振子势
ax1 = fig.add_subplot(221, projection='3d')
plot_3d_surface(ax1, X, Y, Z_harm, '谐振子势能面', 0, 10)
ax1.view_init(elev=30, azim=-45) # 稍微调整一下默认视角

# (2) 三角函数势
ax2 = fig.add_subplot(222, projection='3d')
plot_3d_surface(ax2, X, Y, Z_trig, '三角函数势能面', -2.5, 2.5)
ax2.view_init(elev=35, azim=45)

# (3) Lennard-Jones 势
ax3 = fig.add_subplot(223, projection='3d')
plot_3d_surface(ax3, X_lj, Y_lj, Z_lj, 'Lennard-Jones 势能面', -2, 5)
# 【核心修改点】提高仰角(elev=45)，并旋转方位角(azim=135)，从上方避开排斥壁俯视势阱
ax3.view_init(elev=45, azim=135) 

# (4) 机器学习势
ax4 = fig.add_subplot(224, projection='3d')
plot_3d_surface(ax4, X_lj, Y_lj, Z_ml, '机器学习势能面', -2, 5)
# 同步调整 ML 的视角以便对比
ax4.view_init(elev=45, azim=135) 

plt.tight_layout()
print("绘制完成！")
plt.savefig("PotentialSurfaces_3D.png", dpi=300)
plt.show()