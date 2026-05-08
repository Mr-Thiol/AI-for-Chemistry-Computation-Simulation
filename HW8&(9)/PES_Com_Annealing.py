import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 定义复杂势能面函数
# ==========================================
def E(vars):
    x, y = vars[0], vars[1]
    E_base = 0.1 * (x**2 + y**2)
    E_min1 = -5.0 * np.exp(-((x - 0)**2 + (y - 2)**2) / (2 * 0.8**2)) # 全局极小值 (-5.0)
    E_min2 = -3.0 * np.exp(-((x - 2)**2 + (y + 1)**2) / (2 * 0.6**2)) # 局部极小值 A (-3.0)
    E_min3 = -2.0 * np.exp(-((x + 2)**2 + (y + 1)**2) / (2 * 0.7**2)) # 局部极小值 B (-2.0)
    return E_base + E_min1 + E_min2 + E_min3

# ==========================================
# 2. 增强版模拟退火算法 (加入边界墙)
# ==========================================
def simulated_annealing_fixed(start_pos, T_init=20.0, T_min=0.01, alpha=0.99, step_size=1.0):
    pos = np.array(start_pos, dtype=float)
    current_E = E(pos)
    traj = [pos.copy()]
    T = T_init
    
    best_pos = pos.copy()
    best_E = current_E

    print(f"开始退火... 初始温度: {T_init}, 降温速率: {alpha}")
    
    while T > T_min:
        # 随机产生试探步长
        new_pos = pos + np.random.uniform(-step_size, step_size, size=2)
        
        # ==========================================
        # [核心修复 1]：添加空气墙（边界截断），防止跑出势能面有效范围
        # ==========================================
        new_pos[0] = np.clip(new_pos[0], -4.0, 4.0)
        new_pos[1] = np.clip(new_pos[1], -3.0, 4.0)
        
        new_E = E(new_pos)
        delta_E = new_E - current_E
        
        # Metropolis 准则：能量降低必接受，能量升高按概率接受
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            pos = new_pos
            current_E = new_E
            traj.append(pos.copy())
            
            # 记录探索到的历史最低点
            if current_E < best_E:
                best_E = current_E
                best_pos = pos.copy()
                
        # ==========================================
        # [核心修复 2]：按照 alpha (例如 0.99) 缓慢降温，给予足够时间越过势垒
        # ==========================================
        T *= alpha
        
    return best_pos, np.array(traj)

# ==========================================
# 3. 运行并绘图
# ==========================================
# 这次我们从 B 点出发 (极度容易陷入右下角的深度 -3.0 的坑)
start_pos = [3.0, 0.0]

# 使用极慢的退火速率 (alpha=0.99) 和较高的初始温度 (T_init=20.0)
sa_best_pos, sa_traj = simulated_annealing_fixed(start_pos, T_init=20.0, alpha=0.99, step_size=1.0)

# 退火结束后，执行一次 BFGS "淬火"，精确滑入谷底
res_quench = minimize(E, sa_best_pos, method='BFGS')
final_pos = res_quench.x

print("-" * 30)
print(f"退火前初始位置 (点B): {start_pos}, 能量: {E(start_pos):.3f}")
print(f"退火后锁定区域: {sa_best_pos.round(3)}, 能量: {E(sa_best_pos):.3f}")
print(f"淬火后最终位置: {final_pos.round(3)}, 最终能量: {res_quench.fun:.3f}")
if res_quench.fun < -4.5:
    print("🎉 恭喜！成功跳出局部最优，找到了全局极小值！")
else:
    print("😢 遗憾，还是掉进局部坑了。你可以尝试继续调高 T_init 或将 alpha 调到 0.995。")

# ==========================================
# 4. 绘图代码
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))
x_mesh = np.linspace(-4, 4, 100)
y_mesh = np.linspace(-3, 4, 100)
X, Y = np.meshgrid(x_mesh, y_mesh)
Z = np.array([E([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

contour = ax.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.8)
fig.colorbar(contour, ax=ax, label='能量 E')

# 画轨迹
ax.plot(sa_traj[:, 0], sa_traj[:, 1], '-', color='grey', linewidth=0.5, alpha=0.5, label='随机游走轨迹')
ax.plot(sa_traj[:, 0], sa_traj[:, 1], '.', color='black', markersize=2)

# 标记起点和终点
ax.plot(start_pos[0], start_pos[1], 'y*', markersize=18, markeredgecolor='black', label='初始起点 B')
ax.plot(final_pos[0], final_pos[1], 'r*', markersize=18, markeredgecolor='black', label='最终找到的极小值')

ax.set_title('带边界约束的缓慢模拟退火 (从 B 点出发)')
ax.set_xlabel('坐标 X')
ax.set_ylabel('坐标 Y')
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('sa_fixed_trajectory.png', dpi=300)
print("图片已保存为 sa_fixed_trajectory.png")
plt.show()