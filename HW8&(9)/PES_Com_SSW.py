import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 定义真实势能面 (V_real)
# ==========================================
def V_real(vars):
    x, y = vars[0], vars[1]
    E_base = 0.1 * (x**2 + y**2)
    E_min1 = -5.0 * np.exp(-((x - 0)**2 + (y - 2)**2) / (2 * 0.8**2)) # 全局极小值 (-5.0)
    E_min2 = -3.0 * np.exp(-((x - 2)**2 + (y + 1)**2) / (2 * 0.6**2)) # 局部极小值 A (-3.0)
    E_min3 = -2.0 * np.exp(-((x + 2)**2 + (y + 1)**2) / (2 * 0.7**2)) # 局部极小值 B (-2.0)
    return E_base + E_min1 + E_min2 + E_min3

# ==========================================
# 2. 定义偏置势和总势能
# ==========================================
def V_bias(vars, gaussians):
    """计算沿轨迹叠加的所有高斯偏置势"""
    e_b = 0.0
    for center, w, ds in gaussians:
        dist_sq = (vars[0] - center[0])**2 + (vars[1] - center[1])**2
        e_b += w * np.exp(-dist_sq / (2 * ds**2))
    return e_b

def V_total(vars, gaussians):
    """V_total = 真实势能面 + 历史高斯偏置势"""
    return V_real(vars) + V_bias(vars, gaussians)

# ==========================================
# 3. 核心：全量 SSW 算法
# ==========================================
def full_ssw(start_pos, ssw_steps=5, w=0.8, ds=0.5, step_size=0.4, kT=3.0):
    # 先淬火到当前坑底
    current_min = minimize(V_real, start_pos, method='BFGS').x
    
    found_minima = [current_min.copy()]
    all_gaussians = [] # 用于绘图展示“填坑脚印”
    all_climbs = []    # 记录爬升轨迹
    
    print("=== 开始全量 SSW 搜索 ===")
    print(f"初始锁定极小值: {current_min.round(3)}, 能量: {V_real(current_min):.3f}")

    for step in range(ssw_steps):
        print(f"\n--- SSW 第 {step + 1} 步 ---")
        gaussians = []
        walk_pos = current_min.copy()
        
        # 1. 随机选取一个推动模式 (在2D体系中就是一个随机的单位方向向量)
        direction = np.random.randn(2)
        direction /= np.linalg.norm(direction)
        print(f"  选定爬升方向向量: {direction.round(2)}")
        
        climb_traj = [walk_pos.copy()]
        escaped = False
        
        # 2. 偏置爬升阶段 (Bias-climbing)
        for climb in range(40): # 最大允许填40个高斯
            # 在当前位置铺设高斯偏置势
            gaussians.append((walk_pos.copy(), w, ds))
            
            # 沿着模式方向强制推进一步 (非常关键！不推它就不走)
            walk_pos = walk_pos + step_size * direction
            
            # 在施加了高斯偏置的势能面上进行局部弛豫
            res = minimize(V_total, walk_pos, args=(gaussians,), method='BFGS')
            walk_pos = res.x
            climb_traj.append(walk_pos.copy())
            
            # 越狱检测：去掉偏置势，在真实面上淬火，看是否还能回到原点
            quench_res = minimize(V_real, walk_pos, method='BFGS')
            new_min = quench_res.x
            
            # 如果淬火后的新极小值距离起点大于阈值，说明成功翻越了过渡态！
            if np.linalg.norm(new_min - current_min) > 0.8:
                print(f"  🚀 成功越过势垒！沿途铺设了 {climb+1} 个高斯势。")
                escaped = True
                break
                
        all_gaussians.extend(gaussians)
        all_climbs.append(np.array(climb_traj))
        
        if not escaped:
            print("  ⚠️ 爬升失败，未能翻越势垒。")
            continue
            
        # 3. Metropolis 蒙特卡洛检验
        delta_E = V_real(new_min) - V_real(current_min)
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / kT):
            print(f"  ✅ [接受] 发现新极小值: {new_min.round(3)}, 能量: {V_real(new_min):.3f} (ΔE: {delta_E:.3f})")
            current_min = new_min
        else:
            print(f"  ❌ [拒绝] 能量升高过多且未通过 MC 检验 (ΔE: {delta_E:.3f})，退回原点。")
            
        found_minima.append(current_min.copy())

    return np.array(found_minima), all_climbs, all_gaussians

# ==========================================
# 4. 运行搜索并深度可视化
# ==========================================
start_pos = [-3.0, 0.0] # 依然从最难的局部极小值 B 出发
minima_history, climb_trajs, gaussians = full_ssw(start_pos, ssw_steps=3, w=1.0, ds=0.4, step_size=0.5, kT=5.0)

fig, ax = plt.subplots(figsize=(12, 9))

# 绘制真实势能面等高线
x_mesh = np.linspace(-4, 4, 100)
y_mesh = np.linspace(-3, 4, 100)
X, Y = np.meshgrid(x_mesh, y_mesh)
Z = np.array([V_real([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

contour = ax.contourf(X, Y, Z, levels=35, cmap='coolwarm', alpha=0.8)
fig.colorbar(contour, ax=ax, label='能量 E')

# 【可视化亮点】：画出 SSW 铺设的高斯偏置势“脚印”
for idx, (center, w, ds) in enumerate(gaussians):
    # 用半透明圆圈表示高斯势的覆盖范围 (这里画出 1个 ds 宽度的核心区)
    circle = plt.Circle((center[0], center[1]), ds*0.5, color='white', alpha=0.3, zorder=2)
    ax.add_patch(circle)
    # 绘制高斯势的中心点
    ax.plot(center[0], center[1], 'w+', markersize=4, alpha=0.5)

# 画爬升轨迹
for i, traj in enumerate(climb_trajs):
    ax.plot(traj[:, 0], traj[:, 1], '-', color='black', linewidth=1.5, zorder=3, label='SSW 爬升轨迹' if i==0 else "")
    ax.plot(traj[:, 0], traj[:, 1], '.', color='yellow', markersize=4, zorder=3)

# 画极小值点跳转顺序
for i, pt in enumerate(minima_history):
    if i == 0:
        ax.plot(pt[0], pt[1], 'y*', markersize=20, markeredgecolor='black', zorder=5, label='起点 (局部最优 B)')
    elif i == len(minima_history) - 1:
        ax.plot(pt[0], pt[1], 'r*', markersize=20, markeredgecolor='black', zorder=5, label='终点')
    else:
        ax.plot(pt[0], pt[1], 'g^', markersize=14, markeredgecolor='black', zorder=5, label='中间找到的极小值' if i==1 else "")

    # 画出连接线指示演化顺序
    if i > 0:
        prev_pt = minima_history[i-1]
        ax.annotate('', xy=pt, xytext=prev_pt,
                    arrowprops=dict(facecolor='magenta', shrink=0.05, width=2, headwidth=8, alpha=0.8),
                    zorder=4)

ax.set_title('全量 SSW 算法：沿轨迹铺设高斯势翻越过渡态', fontsize=16)
ax.set_xlabel('坐标 X', fontsize=12)
ax.set_ylabel('坐标 Y', fontsize=12)
ax.legend(loc='lower right')

# 手动添加图例解释高斯圈
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=10, alpha=0.4, lw=0)]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + custom_lines, labels + ['高斯偏置势(脚印)'], loc='lower right')

plt.tight_layout()
plt.savefig('full_ssw_footprints.png', dpi=300)
print("\n绘制完成！请查看 full_ssw_footprints.png")
plt.show()