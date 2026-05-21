import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 设置中文字体（确保Matplotlib能显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 谐振子势 (Harmonic Oscillator)
x_harm = np.linspace(-2, 6, 100)
k = 1.0
r_e = 2.0
y_harm = 0.5 * k * (x_harm - r_e)**2

# 2. 三角函数势 (Trigonometric Function)
x_trig = np.linspace(-2, 6, 100)
y_trig = -np.cos(x_trig - r_e)

# 3. Lennard-Jones势
x_lj = np.linspace(0.85, 4.0, 100)
epsilon = 1.0
sigma = 1.0
# 公式: E = 4ε[(σ/r)^12 - (σ/r)^6]
y_lj = 4 * epsilon * ((sigma/x_lj)**12 - (sigma/x_lj)**6)
r_min_lj = sigma * (2**(1/6))

# 4. 机器学习势 (使用神经网络拟合 LJ 势)
# 生成训练数据
X_train = np.linspace(0.9, 4.0, 40).reshape(-1, 1)
y_train = 4 * epsilon * ((sigma/X_train)**12 - (sigma/X_train)**6).ravel()

# 创建并训练多层感知机(MLP)
mlp = MLPRegressor(hidden_layer_sizes=(20, 20), activation='tanh', solver='lbfgs', max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)

# 预测平滑曲线
x_ml = np.linspace(0.85, 4.0, 100).reshape(-1, 1)
y_ml = mlp.predict(x_ml)

# === 开始绘图 ===
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# 绘制谐振子
axs[0, 0].plot(x_harm, y_harm, 'b-', lw=2)
axs[0, 0].set_title('谐振子势 (二次函数)')
axs[0, 0].set_xlabel('坐标 (r)')
axs[0, 0].set_ylabel('能量 (E)')
axs[0, 0].axvline(r_e, color='r', linestyle='--', alpha=0.5, label='极小值点')
axs[0, 0].legend()

# 绘制三角函数
axs[0, 1].plot(x_trig, y_trig, 'g-', lw=2)
axs[0, 1].set_title('三角函数势')
axs[0, 1].set_xlabel('坐标 (r)')
axs[0, 1].set_ylabel('能量 (E)')
axs[0, 1].axvline(r_e, color='r', linestyle='--', alpha=0.5, label='极小值点')
axs[0, 1].legend()

# 绘制 LJ
axs[1, 0].plot(x_lj, y_lj, 'orange', lw=2)
axs[1, 0].set_title('Lennard-Jones势')
axs[1, 0].set_xlabel('距离 (r)')
axs[1, 0].set_ylabel('能量 (E)')
axs[1, 0].set_ylim(-1.5, 3)
axs[1, 0].axvline(r_min_lj, color='r', linestyle='--', alpha=0.5, label='极小值点 (Re)')
axs[1, 0].legend()

# 绘制 ML
axs[1, 1].plot(x_ml, y_ml, 'purple', lw=2, label='机器学习预测')
axs[1, 1].plot(X_train, y_train, 'k.', markersize=6, alpha=0.5, label='训练数据 (LJ)')
axs[1, 1].set_title('机器学习势函数 (拟合LJ势)')
axs[1, 1].set_xlabel('距离 (r)')
axs[1, 1].set_ylabel('能量 (E)')
axs[1, 1].set_ylim(-1.5, 3)
axs[1, 1].axvline(r_min_lj, color='r', linestyle='--', alpha=0.5, label='极小值点')
axs[1, 1].legend()

plt.tight_layout()
plt.savefig("PotentialSurfaces_2D.png", dpi=300)
plt.show()