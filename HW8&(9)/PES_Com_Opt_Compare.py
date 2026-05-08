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
    # 极小值1(全局): 坐标大概在(0, 2)，深度 -5.0
    E_min1 = -5.0 * np.exp(-((x - 0)**2 + (y - 2)**2) / (2 * 0.8**2))
    # 极小值2(局部A): 坐标大概在(2, -1)，深度 -3.0
    E_min2 = -3.0 * np.exp(-((x - 2)**2 + (y + 1)**2) / (2 * 0.6**2))
    # 极小值3(局部B): 坐标大概在(-2, -1)，深度 -2.0
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
def optimize_gd(start_pos, lr=0.15, max_iter=200, tol=1e-4):
    pos = np.array(start_pos, dtype=float)
    traj = [pos.copy()]
    for _ in range(max_iter):
        grad = compute_gradient(E, pos)
        if np.linalg.norm(grad) < tol: break
        pos = pos - lr * grad
        traj.append(pos.copy())
    return np.array(traj)

def optimize_adam(start_pos, lr=0.15, max_iter=200, tol=1e-4):
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
# 3. 准备绘图网格和初始点
# ==========================================
x_mesh = np.linspace(-4, 4, 100)
y_mesh = np.linspace(-3, 4, 100)
X, Y = np.meshgrid(x_mesh, y_mesh)
Z = np.array([E([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

# 定义三个不同的初始点
initial_points = [
    ([0.0, 3.5], "初始点 A\n(目标：全局极小值)"),
    ([3.0, 0.0], "初始点 B\n(目标：局部极小值 A)"),
    ([-3.0, 0.0], "初始点 C\n(目标：局部极小值 B)")
]

# 创建 1x3 的子图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('不同初始点对局部优化算法收敛结果的决定性影响', fontsize=18, y=1.02)

# ==========================================
# 4. 循环在三个初始点上运行并绘图
# ==========================================
for idx, (start_pos, title) in enumerate(initial_points):
    ax = axs[idx]
    
    # 获取轨迹
    traj_gd = optimize_gd(start_pos)
    traj_adam = optimize_adam(start_pos)
    
    traj_bfgs = [np.array(start_pos)]
    minimize(E, start_pos, method='BFGS', callback=lambda xk: traj_bfgs.append(xk.copy()))
    traj_bfgs = np.array(traj_bfgs)

    traj_cg = [np.array(start_pos)]
    minimize(E, start_pos, method='CG', callback=lambda xk: traj_cg.append(xk.copy()))
    traj_cg = np.array(traj_cg)
    
    # 画地形图
    contour = ax.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.8)
    
    # 画四种算法的轨迹
    ax.plot(traj_gd[:, 0], traj_gd[:, 1], 'o-', color='black', markersize=3, linewidth=1, label='GD')
    ax.plot(traj_adam[:, 0], traj_adam[:, 1], 's-', color='purple', markersize=3, linewidth=1, label='ADAM')
    ax.plot(traj_bfgs[:, 0], traj_bfgs[:, 1], '^-', color='red', markersize=4, linewidth=1.5, label='BFGS')
    ax.plot(traj_cg[:, 0], traj_cg[:, 1], 'x-', color='green', markersize=4, linewidth=1.5, label='CG')
    
    # 标记起点
    ax.plot(start_pos[0], start_pos[1], 'y*', markersize=18, markeredgecolor='black', label='初始起点')
    
    # 设置图表属性
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('坐标 X')
    ax.set_ylabel('坐标 Y')
    if idx == 0:
        ax.legend(loc='lower left')

# 添加一个统一的 Colorbar
fig.colorbar(contour, ax=axs.ravel().tolist(), shrink=0.8, label='能量 E')

plt.savefig('initial_points_comparison.png', dpi=300, bbox_inches='tight')
print("图片已成功生成并保存为 initial_points_comparison.png")
# plt.show()