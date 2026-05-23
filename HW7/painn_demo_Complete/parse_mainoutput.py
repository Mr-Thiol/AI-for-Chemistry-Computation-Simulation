"""
解析 Colab 训练输出文本 → log/train_log.csv

用法：
    python parse_mainoutput.py                   # 默认解析 mainoutput.txt
    python parse_mainoutput.py myfile.txt        # 指定文件

输出：log/train_log.csv（可直接供 visualize.py 使用）
"""

import os
import re
import csv
import sys

BASE   = os.path.dirname(os.path.abspath(__file__))
LOG    = os.path.join(BASE, 'log')
IN_TXT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'mainoutput.txt')
OUT_CSV = os.path.join(LOG, 'train_log.csv')

# ── 正则表达式 ──────────────────────────────────────────────
# [EPOCH] 120 [RMSE] [energy] [mol] 1.025823 [atom] 0.077435 [force] 0.419512 [time] 0:00:06.811434
RE_RMSE = re.compile(
    r'\[EPOCH\]\s+(\d+)\s+\[RMSE\].*?\[mol\]\s+([\d.]+).*?\[atom\]\s+([\d.]+).*?\[force\]\s+([\d.]+).*?\[time\]\s+([\d:\.]+)'
)
# [EPOCH] 120 [MAE]  [energy] [mol] 0.856541 [atom] -------- [force] 0.146186
RE_MAE = re.compile(
    r'\[EPOCH\]\s+(\d+)\s+\[MAE\].*?\[mol\]\s+([\d.]+).*?\[force\]\s+([\d.]+)'
)

# ── 解析 ───────────────────────────────────────────────────
epochs = {}   # epoch_id -> {'train': {...}, 'valid': {...}}

with open(IN_TXT, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        m_rmse = RE_RMSE.search(line)
        m_mae  = RE_MAE.search(line)

        if m_rmse:
            ep   = int(m_rmse.group(1))
            data = {
                'e_rmse_mol':  float(m_rmse.group(2)),
                'e_rmse_atom': float(m_rmse.group(3)),
                'f_rmse':      float(m_rmse.group(4)),
                'time':        m_rmse.group(5),
            }
            rec = epochs.setdefault(ep, {'rmse': [], 'mae': []})
            rec['rmse'].append(data)

        elif m_mae:
            ep   = int(m_mae.group(1))
            data = {
                'e_mae': float(m_mae.group(2)),
                'f_mae': float(m_mae.group(3)),
            }
            rec = epochs.setdefault(ep, {'rmse': [], 'mae': []})
            rec['mae'].append(data)

# ── 写 CSV ─────────────────────────────────────────────────
os.makedirs(LOG, exist_ok=True)

# 判断 CSV 是否已有内容（接续模式）
existing_epochs = set()
append_mode = os.path.exists(OUT_CSV)
if append_mode:
    with open(OUT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_epochs.add(int(row['epoch']))
    print(f"[接续] 已有 CSV 包含 {len(existing_epochs)} 条记录，"
          f"epoch 范围：{min(existing_epochs)}–{max(existing_epochs)}")

HEADER = ['epoch', 'phase', 'e_rmse_mol', 'e_rmse_atom', 'f_rmse', 'e_mae', 'f_mae']

rows_written = 0
skipped = 0
with open(OUT_CSV, 'a' if append_mode else 'w', newline='') as f:
    writer = csv.writer(f)
    if not append_mode:
        writer.writerow(HEADER)

    for ep in sorted(epochs.keys()):
        if ep in existing_epochs:
            skipped += 1
            continue

        rec = epochs[ep]
        rmse_list = rec.get('rmse', [])
        mae_list  = rec.get('mae',  [])

        # 第 1 对 → train，第 2 对 → valid
        for phase_idx, phase in enumerate(('train', 'valid')):
            if phase_idx >= len(rmse_list) or phase_idx >= len(mae_list):
                continue
            r = rmse_list[phase_idx]
            m = mae_list[phase_idx]
            writer.writerow([ep, phase,
                             r['e_rmse_mol'], r['e_rmse_atom'], r['f_rmse'],
                             m['e_mae'],      m['f_mae']])
            rows_written += 1

print(f"[完成] 写入 {rows_written} 行，跳过已有 epoch {skipped} 个")
print(f"[输出] {OUT_CSV}")

# ── 统计 ───────────────────────────────────────────────────
all_epochs = sorted(epochs.keys())
if all_epochs:
    print(f"[统计] 本文件覆盖 epoch {min(all_epochs)}–{max(all_epochs)}"
          f"（共 {len(all_epochs)} 个 epoch）")
