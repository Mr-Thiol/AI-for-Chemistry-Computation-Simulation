import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 定义势能面函数与梯度
# ==========================================
def E(vars):
    x, y = vars[0], vars[1]
    E_base = 0.1 * (x**2 + y**2)
    E_min1 = -5.0 * np.exp(-((x - 0)**2 + (y - 2)**2) / (2 * 0.8**2))
    E_min2 = -3.0 * np.exp(-((x - 2)**2 + (y + 1)**2) / (2 * 0.6**2))
    E_min3 = -2.0 * np.exp(-((x + 2)**2 + (y + 1)**2) / (2 * 0.7**2))
    return E_base + E_min1 + E_min2 + E_min3

def compute_gradient(func, vars, eps=1e-5):
    x, y = vars
    df_dx = (func([x + eps, y]) - func([x - eps, y])) / (2 * eps)
    df_dy = (func([x, y + eps]) - func([x, y - eps])) / (2 * eps)
    return np.array([df_dx, df_dy])

# ==========================================
# 2. 自定义优化算法 (记录轨迹)
# ==========================================
def optimize_gd(start_pos, lr=0.1, max_iter=200, tol=1e-4):
    pos = np.array(start_pos, dtype=float)
    traj = [pos.copy()]
    for _ in range(max_iter):
        grad = compute_gradient(E, pos)
        if np.linalg.norm(grad) < tol: break
        pos = pos - lr * grad
        traj.append(pos.copy())
    return np.array(traj)

def optimize_adam(start_pos, lr=0.1, max_iter=200, tol=1e-4):
    pos = np.array(start_pos, dtype=float)
    traj = [pos.copy()]
    m, v = np.zeros(2), np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, max_iter + 1):
        grad = compute_gradient(E, pos)
        if np.linalg.norm(grad) < tol: break
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        pos = pos - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(pos.copy())
    return np.array(traj)

# ==========================================
# 3. 运行算法并收集轨迹
# ==========================================
# 选择一个初始点，例如靠近局部极小值 A 的边界
start_pos = [3.5, 1.0]

# GD 和 ADAM
traj_gd = optimize_gd(start_pos, lr=0.15)
traj_adam = optimize_adam(start_pos, lr=0.15)

# BFGS 和 CG (利用 scipy 的 callback 提取轨迹)
traj_bfgs = [np.array(start_pos)]
minimize(E, start_pos, method='BFGS', callback=lambda xk: traj_bfgs.append(xk.copy()))
traj_bfgs = np.array(traj_bfgs)

traj_cg = [np.array(start_pos)]
minimize(E, start_pos, method='CG', callback=lambda xk: traj_cg.append(xk.copy()))
traj_cg = np.array(traj_cg)

# ==========================================
# 4. 绘图：等高线地形图 + 损失下降曲线
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---- 左图：等高线与移动轨迹 ----
x_mesh = np.linspace(-4, 4, 100)
y_mesh = np.linspace(-3, 4, 100)
X, Y = np.meshgrid(x_mesh, y_mesh)
Z = np.array([E([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

# 画等高线
contour = ax1.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.8)
fig.colorbar(contour, ax=ax1, shrink=0.8, label='能量 E')

# 画轨迹
algorithms = [
    ('GD', traj_gd, 'black', 'o-'),
    ('ADAM', traj_adam, 'purple', 's-'),
    ('BFGS', traj_bfgs, 'red', '^-'),
    ('CG', traj_cg, 'green', 'x-')
]

for name, traj, color, fmt in algorithms:
    ax1.plot(traj[:, 0], traj[:, 1], fmt, color=color, markersize=4, linewidth=1.5, label=name)

# 标记起点
ax1.plot(start_pos[0], start_pos[1], 'y*', markersize=15, markeredgecolor='black', label='初始起点')
ax1.set_title('等高线图：各算法“下山”轨迹对比')
ax1.set_xlabel('坐标 X')
ax1.set_ylabel('坐标 Y')
ax1.legend()

# ---- 右图：能量 (Loss) 随步数的下降曲线 ----
for name, traj, color, _ in algorithms:
    energy_history = [E(pos) for pos in traj]
    # 限制显示前50步，以便看清前期差异
    steps = min(50, len(energy_history)) 
    ax2.plot(range(steps), energy_history[:steps], color=color, linewidth=2, label=name)

ax2.set_title('能量下降曲线 (Loss Curve)')
ax2.set_xlabel('迭代步数 (Iterations)')
ax2.set_ylabel('能量 E')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig('optimization_paths.png', dpi=300)
print("可视化完成，已保存为 optimization_paths.png")
plt.show()