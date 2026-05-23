"""
可视化 PaiNN 训练与模拟结果
  - log/opt.arc      → BFGS 优化能量收敛 + 最终结构 3D 图
  - log/md.log       → NVE 分子动力学能量与温度演化
  - log/train_log.csv → 训练损失曲线（需先运行修改后的 main.py 生成）

依赖：matplotlib, numpy（无需 ASE / PyTorch）
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
LOG  = os.path.join(BASE, 'log')

ELEM_COLOR = {'Ti': '#5BA4CF', 'O': '#FF3B30'}
ELEM_SIZE  = {'Ti': 120,       'O': 55}

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


# ══════════════════════════════════════════════════════════════
# 工具：解析 BIOSYM .arc 文件
# ══════════════════════════════════════════════════════════════
def parse_arc(filepath):
    """
    返回 frames 列表，每个 frame 是：
        {'step': int, 'energy': float,
         'cell': [a,b,c,α,β,γ],
         'atoms': [(elem, x, y, z), ...]}
    """
    frames, current = [], None
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            # 帧头：Energy   <step>   0.0000   <energy>
            if parts[0] == 'Energy' and len(parts) == 4:
                if current is not None:
                    frames.append(current)
                current = {
                    'step':   int(parts[1]),
                    'energy': float(parts[3]),
                    'atoms':  [],
                    'cell':   None,
                }
            # 晶胞：PBC   a b c α β γ
            elif parts[0] == 'PBC' and len(parts) == 7 and current is not None:
                current['cell'] = [float(x) for x in parts[1:]]
            # 原子行：Elem   x   y   z   CORE   ...
            elif len(parts) >= 5 and parts[4] == 'CORE' and current is not None:
                elem = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                current['atoms'].append((elem, x, y, z))
    if current is not None:
        frames.append(current)
    return frames


# ══════════════════════════════════════════════════════════════
# Figure 1：BFGS 结构优化
# ══════════════════════════════════════════════════════════════
def plot_optimization():
    path = os.path.join(LOG, 'opt.arc')
    if not os.path.exists(path):
        print(f"[跳过] 未找到 {path}")
        return

    frames = parse_arc(path)
    if not frames:
        print("[跳过] opt.arc 中无有效帧")
        return

    steps    = [f['step']   for f in frames]
    energies = [f['energy'] for f in frames]
    initial  = frames[0]
    final    = frames[-1]

    fig = plt.figure(figsize=(14, 8), layout='constrained')
    fig.suptitle('BFGS Structure Optimization  (TiOx · PaiNN)', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # ── 子图1：能量收敛曲线（跨全行） ───────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps, energies, color='#2878B5', lw=2, marker='o', ms=4)
    ax1.axhline(y=energies[-1], color='gray', ls='--', lw=1, alpha=0.5)

    e_range = max(energies) - min(energies) if max(energies) != min(energies) else 1.0
    ax1.annotate(f'E₀ = {energies[0]:.4f} eV',
                 xy=(steps[0], energies[0]),
                 xytext=(steps[0] + max(1, len(steps)*0.05),
                         energies[0] + e_range * 0.06),
                 fontsize=9, color='#E84B23',
                 arrowprops=dict(arrowstyle='->', color='#E84B23', lw=1))
    ax1.annotate(f'E_final = {energies[-1]:.4f} eV',
                 xy=(steps[-1], energies[-1]),
                 xytext=(steps[-1] - max(3, len(steps)*0.3),
                         energies[-1] - e_range * 0.12),
                 fontsize=9, color='#2878B5',
                 arrowprops=dict(arrowstyle='->', color='#2878B5', lw=1))

    delta_e = energies[-1] - energies[0]
    ax1.set_xlabel('BFGS Step')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title(f'Energy Convergence   ΔE = {delta_e:.4f} eV'
                  f'   ({len(steps)} steps → {len(steps)-1} iterations)',
                  fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── 子图2-4：最终结构三视图（正/侧/俯） ─────────────
    n_ti = sum(1 for a in final['atoms'] if a[0] == 'Ti')
    n_o  = sum(1 for a in final['atoms'] if a[0] == 'O')
    cell = final['cell']
    cell_str = (f'a={cell[0]:.2f}  b={cell[1]:.2f}  c={cell[2]:.2f} Å'
                if cell else '')

    # 三视图定义：(title, xlabel, ylabel, x_idx, y_idx)
    views = [
        ('Front View  (XZ)', 'x (Å)', 'z (Å)', 0, 2),
        ('Side View   (YZ)', 'y (Å)', 'z (Å)', 1, 2),
        ('Top View    (XY)', 'x (Å)', 'y (Å)', 0, 1),
    ]

    for col_idx, (title, xlabel, ylabel, xi, yi) in enumerate(views):
        ax = fig.add_subplot(gs[1, col_idx])
        for elem in ('Ti', 'O'):
            pts = [(a[xi+1], a[yi+1]) for a in final['atoms'] if a[0] == elem]
            # a[1]=x, a[2]=y, a[3]=z  →  a[xi+1], a[yi+1]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys,
                           c=ELEM_COLOR[elem], s=ELEM_SIZE[elem],
                           label=f'{elem} ({len(pts)})',
                           alpha=0.75, edgecolors='white', linewidths=0.3)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.25)
        if col_idx == 0:
            ax.legend(fontsize=8, loc='upper right',
                      title=f'{n_ti+n_o} atoms\n{cell_str}',
                      title_fontsize=7)

    out = os.path.join(LOG, 'opt_visualization.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'[保存] {out}')
    plt.show()



# ══════════════════════════════════════════════════════════════
# Figure 2：NVE 分子动力学
# ══════════════════════════════════════════════════════════════
def plot_md():
    path = os.path.join(LOG, 'md.log')
    if not os.path.exists(path):
        print(f"[跳过] 未找到 {path}")
        return

    # 解析 md.log：Time Etot Epot Ekin T
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # 跳过表头
            parts = line.split()
            if len(parts) == 5:
                data.append([float(x) for x in parts])
    if not data:
        print("[跳过] md.log 无有效数据")
        return

    arr  = np.array(data)
    time = arr[:, 0]
    etot = arr[:, 1]
    epot = arr[:, 2]
    ekin = arr[:, 3]
    temp = arr[:, 4]

    fig = plt.figure(figsize=(14, 8), layout='constrained')
    fig.suptitle('NVE Molecular Dynamics  (TiOx · PaiNN)', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # ── 行0全宽：温度曲线 ────────────────────────────────
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.plot(time, temp, color='#FA8C35', lw=1.5, alpha=0.9)
    ax_t.axhline(y=temp.mean(), color='gray', ls='--', lw=1,
                 label=f'<T> = {temp.mean():.1f} K')
    ax_t.fill_between(time, temp.mean() - temp.std(), temp.mean() + temp.std(),
                      alpha=0.15, color='#FA8C35')
    ax_t.set_xlabel('Time (ps)')
    ax_t.set_ylabel('Temperature (K)')
    ax_t.set_title(f'Temperature  T_avg = {temp.mean():.1f} ± {temp.std():.1f} K',
                   fontsize=10)
    ax_t.legend(fontsize=9)
    ax_t.grid(True, alpha=0.3)

    # ── 行1左：Etot（y 轴紧致，验证 NVE 守恒） ──────────
    ax_et = fig.add_subplot(gs[1, 0])
    ax_et.plot(time, etot, color='#2878B5', lw=1.5)
    et_margin = max(etot.std() * 5, 1e-3)
    ax_et.set_ylim(etot.mean() - et_margin, etot.mean() + et_margin)
    ax_et.set_xlabel('Time (ps)')
    ax_et.set_ylabel('E_tot (eV)')
    ax_et.set_title(f'E_tot  (NVE conservation)\nfluctuation: {etot.std()*1000:.2f} meV',
                    fontsize=9)
    ax_et.grid(True, alpha=0.3)

    # ── 行1中：Epot（各自 y 轴，看波动） ─────────────────
    ax_ep = fig.add_subplot(gs[1, 1])
    ax_ep.plot(time, epot, color='#E84B23', lw=1.5)
    ep_margin = max(epot.std() * 5, 0.05)
    ax_ep.set_ylim(epot.mean() - ep_margin, epot.mean() + ep_margin)
    ax_ep.set_xlabel('Time (ps)')
    ax_ep.set_ylabel('E_pot (eV)')
    ax_ep.set_title(f'E_pot  mean={epot.mean():.3f} eV\nstd={epot.std()*1000:.1f} meV',
                    fontsize=9)
    ax_ep.grid(True, alpha=0.3)

    # ── 行1右：Ekin（应与温度同步波动） ──────────────────
    ax_ek = fig.add_subplot(gs[1, 2])
    ax_ek.plot(time, ekin, color='#28B57A', lw=1.5)
    ek_margin = max(ekin.std() * 5, 0.05)
    ax_ek.set_ylim(max(0.0, ekin.mean() - ek_margin), ekin.mean() + ek_margin)
    ax_ek.set_xlabel('Time (ps)')
    ax_ek.set_ylabel('E_kin (eV)')
    ax_ek.set_title(f'E_kin  mean={ekin.mean():.3f} eV\nstd={ekin.std()*1000:.1f} meV',
                    fontsize=9)
    ax_ek.grid(True, alpha=0.3)

    out = os.path.join(LOG, 'md_visualization.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'[保存] {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════
# Figure 3：训练损失（需要 main.py 生成 train_log.csv）
# ══════════════════════════════════════════════════════════════
def plot_training_loss():
    path = os.path.join(LOG, 'train_log.csv')
    if not os.path.exists(path):
        print(f"[跳过] 未找到 {path}")
        print("  → 请先用修改后的 main.py 训练一次，会自动生成 log/train_log.csv")
        return

    # 读取 CSV：epoch,phase,e_rmse_mol,e_rmse_atom,f_rmse,e_mae,f_mae
    import csv
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (int(v) if k in ('epoch', 'phase') and v.isdigit()
                             else v)
                         for k, v in row.items()})

    train_rows = [r for r in rows if r['phase'] == 'train']
    valid_rows = [r for r in rows if r['phase'] == 'valid']

    def col(rows, key):
        return np.array([float(r[key]) for r in rows])

    epochs_t = col(train_rows, 'epoch')
    epochs_v = col(valid_rows, 'epoch')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('PaiNN Training Loss  (TiOx)', fontsize=14, fontweight='bold')
    labels = [
        ('e_rmse_atom', 'Energy RMSE / atom (eV)',  'Energy RMSE'),
        ('f_rmse',      'Force RMSE (eV/Å)',        'Force RMSE'),
        ('e_mae',       'Energy MAE / mol (eV)',    'Energy MAE'),
        ('f_mae',       'Force MAE (eV/Å)',         'Force MAE'),
    ]

    for ax, (key, ylabel, title) in zip(axes.flat, labels):
        ax.plot(epochs_t, col(train_rows, key), '#2878B5', lw=1.5, label='Train')
        if valid_rows:
            ax.plot(epochs_v, col(valid_rows, key), '#E84B23', lw=1.5,
                    ls='--', label='Valid')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # 对数坐标（损失跨越多个数量级时更直观）
        if col(train_rows, key).max() / (col(train_rows, key).min() + 1e-10) > 10:
            ax.set_yscale('log')

    plt.tight_layout()
    out = os.path.join(LOG, 'train_loss_visualization.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'[保存] {out}')
    plt.show()


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(LOG, exist_ok=True)

    print('=' * 55)
    print('  PaiNN 结果可视化')
    print('=' * 55)

    print('\n[1/3] BFGS 结构优化 ...')
    plot_optimization()

    print('\n[2/3] 分子动力学 ...')
    plot_md()

    print('\n[3/3] 训练损失 ...')
    plot_training_loss()

    print('\n完成。图片已保存至 log/ 文件夹。')
