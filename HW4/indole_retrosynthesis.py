"""
indole_retrosynthesis.py
吲哚 (Indole) 逆合成分析一体化脚本
——预测 + 3D 结构优化 + 格式转换 + 2D/3D 可视化

运行方式：
    conda activate chemformer
    cd E:\\JupyterPjs\\AI4Chem\\HW4
    python indole_retrosynthesis.py
"""

import os
import sys
import json
import math
import time
from io import BytesIO

# ── 路径 ──────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CHEMFORMER    = os.path.join(SCRIPT_DIR, "Chemformer")
sys.path.insert(0, CHEMFORMER)

CKPT_ORIG     = os.path.join(SCRIPT_DIR, "fine_tune_upsto_50_last.ckpt")
CKPT_V2       = os.path.join(SCRIPT_DIR, "fine_tune_upsto_50_last_v2.ckpt")
VOCAB_PATH    = os.path.join(CHEMFORMER, "bart_vocab_downstream.json")

# 所有输出文件以 indole_ 开头
OUT_DIR       = os.path.join(SCRIPT_DIR, "mol_analysis_indole")
OPT_DIR       = os.path.join(OUT_DIR, "optimized")
IMG_DIR       = os.path.join(OUT_DIR, "images")
INDOLE_CSV    = os.path.join(OUT_DIR, "indole_result.csv")
PROP_CSV      = os.path.join(OUT_DIR, "indole_mol_properties.csv")
HTML_PATH     = os.path.join(OUT_DIR, "indole_viewer_3d.html")

for d in [OUT_DIR, OPT_DIR, IMG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 目标分子：吲哚 ─────────────────────────────────────────────────
PRODUCT_SMILES = "c1ccc2[nH]ccc2c1"   # 吲哚 Indole，MW=117.15
PRODUCT_NAME   = "Indole (吲哚)"
N_BEAMS        = 10    # 多取几条路径，吲哚是基础骨架合成路线丰富
N_CONFS        = 50    # 构象搜索数量

# ══════════════════════════════════════════════════════════════════
# 步骤 1：修复 checkpoint 并加载模型
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 1：加载 Chemformer 模型")
print("=" * 60)

import torch
if not os.path.exists(CKPT_V2):
    print("修复 checkpoint（vocab_size → vocabulary_size）...")
    ckpt = torch.load(CKPT_ORIG, map_location="cpu")
    hp   = ckpt.get("hyper_parameters", {})
    if "vocab_size" in hp and "vocabulary_size" not in hp:
        hp["vocabulary_size"] = hp.pop("vocab_size")
        ckpt["hyper_parameters"] = hp
        torch.save(ckpt, CKPT_V2)
        print(f"  已保存修复后的 checkpoint: {os.path.basename(CKPT_V2)}")
    else:
        CKPT_V2 = CKPT_ORIG
        print("  checkpoint 键名已正常，无需修复。")
else:
    print("  使用已修复的 checkpoint。")

import molbart.utils.data_utils as util
from omegaconf import OmegaConf
from molbart.models import Chemformer
from molbart.data import SynthesisDataModule
import pandas as pd

config = OmegaConf.create({
    "train_mode":            "eval",
    "batch_size":            1,
    "n_gpus":                0,
    "n_beams":               N_BEAMS,
    "n_unique_beams":        None,
    "vocabulary_path":       VOCAB_PATH,
    "model_path":            CKPT_V2,
    "model_type":            "bart",
    "task":                  "backward_prediction",
    "data_path":             None,
    "dataset_part":          "full",
    "i_chunk":               0,
    "n_chunks":              1,
    "datamodule":            None,
    "scorers":               None,
    "output_sampled_smiles": None,
})

chemformer = Chemformer(config)
print("  模型加载完成！\n")

# ══════════════════════════════════════════════════════════════════
# 步骤 2：逆合成预测
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 2：逆合成预测（吲哚，{} beams）".format(N_BEAMS))
print("=" * 60)

datamodule = SynthesisDataModule(
    reactants=[PRODUCT_SMILES],
    products=[PRODUCT_SMILES],
    tokenizer=chemformer.tokenizer,
    batch_size=1,
    max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
    dataset_path=""
)
datamodule.setup()

t0 = time.time()
smiles_beams, log_lhs_beams, _ = chemformer.predict(
    dataloader=datamodule.full_dataloader()
)
elapsed = time.time() - t0
print(f"\n预测完成，耗时 {elapsed:.1f} 秒\n")

# 解析结果
from rdkit import Chem

rows = []
print(f"{'='*60}")
print(f"目标分子: {PRODUCT_NAME}  ({PRODUCT_SMILES})")
print(f"{'='*60}")
for i, (smi, lh) in enumerate(zip(smiles_beams[0], log_lhs_beams[0])):
    parts = smi.split(".")
    # 验证 SMILES 有效性
    valid = all(Chem.MolFromSmiles(p) is not None for p in parts)
    tag   = "✓" if valid else "✗ 无效"
    print(f"  Beam {i+1:2d} [{tag}]: {smi}")
    print(f"           log-likelihood = {lh:.4f}\n")
    rows.append({
        "beam":             i + 1,
        "reactants_smiles": smi,
        "log_likelihood":   lh,
        "n_reactants":      len(parts),
        "reactant_1":       parts[0] if len(parts) > 0 else "",
        "reactant_2":       parts[1] if len(parts) > 1 else "",
        "valid_smiles":     valid,
    })

df = pd.DataFrame(rows)
df.to_csv(INDOLE_CSV, index=False)
print(f"结果已保存: {INDOLE_CSV}\n")

# 仅保留有效 SMILES 的 beams 做后续分析
df_valid = df[df["valid_smiles"]].reset_index(drop=True)
print(f"有效 beam 数: {len(df_valid)}/{len(df)}\n")

# ══════════════════════════════════════════════════════════════════
# 步骤 3：3D 结构优化 + 格式转换
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 3：3D 结构优化（多构象 + MMFF94）")
print("=" * 60)

from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdDepictor

def optimize_3d(mol, name, n_confs=N_CONFS):
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params)
    if len(conf_ids) == 0:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol_h)
        return mol_h, None, 0
    ff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
    energies = []
    for cid in conf_ids:
        if ff_props:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, ff_props, confId=cid)
            if ff:
                ff.Minimize(maxIts=2000)
                energies.append((ff.CalcEnergy(), cid))
            else:
                energies.append((float("inf"), cid))
        else:
            AllChem.UFFOptimizeMolecule(mol_h, confId=cid)
            uff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
            energies.append((uff.CalcEnergy() if uff else float("inf"), cid))
    energies.sort(key=lambda x: x[0])
    best_e, best_cid = energies[0]
    # 移除非最优构象
    mol_best = mol_h
    for _, cid in energies[1:]:
        try: mol_best.RemoveConformer(cid)
        except: pass
    return mol_best, best_e, len(conf_ids)

# 收集唯一分子
mol_registry = {}   # canon_smiles → {name, mol}

def register_mol(smiles, name):
    if not smiles or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles.strip())
    if m is None:
        return None
    canon = Chem.MolToSmiles(m)
    if canon not in mol_registry:
        mol_registry[canon] = {"name": name, "mol": m}
    return canon

target_canon = register_mol(PRODUCT_SMILES, PRODUCT_NAME)
for _, row in df_valid.iterrows():
    b = int(row["beam"])
    register_mol(str(row["reactant_1"]), f"Beam {b} / 反应物 1")
    if str(row["reactant_2"]).strip():
        register_mol(str(row["reactant_2"]), f"Beam {b} / 反应物 2")

print(f"共 {len(mol_registry)} 个唯一分子\n")

props_rows = []
optimized_mols = {}   # canon → (opt_mol, sdf_block)

for canon, info in mol_registry.items():
    mol  = info["mol"]
    name = info["name"]
    safe = "indole_" + name.replace("/", "-").replace(" ", "_")
    print(f"  {name}  SMILES={canon}")

    opt_mol, energy, n_ok = optimize_3d(mol, name)
    e_str = f"{energy:.2f} kcal/mol" if energy is not None else "N/A"
    print(f"    ✓ {n_ok}/{N_CONFS} 构象  最低能量={e_str}")

    # 性质计算
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings= rdMolDescriptors.CalcNumRings(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    props_rows.append({
        "name": name, "smiles": canon,
        "MW": round(mw, 2), "logP": round(logp, 2),
        "HBD": hbd, "HBA": hba, "TPSA": round(tpsa, 1),
        "RotBonds": rotb, "Rings": rings, "AromaticRings": arom,
        "n_confs_ok": n_ok,
        "MMFF94_energy_kcal": round(energy, 3) if energy is not None else "N/A",
    })

    # SDF
    writer = Chem.SDWriter(os.path.join(OPT_DIR, f"{safe}.sdf"))
    opt_mol.SetProp("_Name", name)
    opt_mol.SetProp("SMILES", canon)
    if energy is not None:
        opt_mol.SetProp("MMFF94_Energy_kcal", str(round(energy, 4)))
    writer.write(opt_mol); writer.close()

    # XYZ
    conf = opt_mol.GetConformer()
    lines = [str(opt_mol.GetNumAtoms()), f"{name}  E={e_str}"]
    for atom in opt_mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<3s}  {p.x:12.6f}  {p.y:12.6f}  {p.z:12.6f}")
    with open(os.path.join(OPT_DIR, f"{safe}.xyz"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # PDB
    Chem.MolToPDBFile(opt_mol, os.path.join(OPT_DIR, f"{safe}.pdb"))

    optimized_mols[canon] = (opt_mol, Chem.MolToMolBlock(opt_mol))

props_df = pd.DataFrame(props_rows)
props_df.to_csv(PROP_CSV, index=False)
print(f"\n性质表已保存: {PROP_CSV}\n")

# ══════════════════════════════════════════════════════════════════
# 步骤 4：2D 结构网格图 + 反应路径图
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 4：2D 可视化")
print("=" * 60)

from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont

MOL_W, MOL_H = 360, 280
FONT_H        = 22
COLS          = 3
BG            = (255, 255, 255)

def smiles_to_png(mol, width=MOL_W, height=MOL_H, legend=""):
    rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(width, height)
    d.drawOptions().padding = 0.12
    d.DrawMolecule(mol, legend=legend)
    d.FinishDrawing()
    return Image.open(BytesIO(d.GetDrawingText()))

try:
    font_sm = ImageFont.truetype("arial.ttf", 13)
    font_md = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font_sm = font_md = ImageFont.load_default()

# ── 网格图 ─────────────────────────────────────────────────────
entries  = [(target_canon, PRODUCT_NAME)] + [
    (c, i["name"]) for c, i in mol_registry.items() if c != target_canon
]
n_mols   = len(entries)
n_rows   = math.ceil(n_mols / COLS)
grid_img = Image.new("RGB", (COLS * MOL_W, n_rows * (MOL_H + FONT_H)), BG)
d_grid   = ImageDraw.Draw(grid_img)

for idx, (canon, name) in enumerate(entries):
    col = idx % COLS; row = idx // COLS
    x   = col * MOL_W; y   = row * (MOL_H + FONT_H)
    mw  = next((r["MW"] for r in props_rows if r["smiles"] == canon), "?")
    img = smiles_to_png(mol_registry[canon]["mol"], legend=f"MW={mw}")
    grid_img.paste(img, (x, y))
    d_grid.rectangle([x, y + MOL_H, x + MOL_W, y + MOL_H + FONT_H], fill=(245, 245, 250))
    d_grid.text((x + 6, y + MOL_H + 4), name[:38], font=font_sm, fill=(50, 50, 80))
    d_grid.rectangle([x, y, x + MOL_W - 1, y + MOL_H + FONT_H - 1], outline=(210, 210, 210))

grid_path = os.path.join(IMG_DIR, "indole_molecules_grid.png")
grid_img.save(grid_path)
print(f"  ✓ 网格图: {grid_path}")

# ── 反应路径图 ─────────────────────────────────────────────────
RXNW, RXNH = 260, 200
PAD, ARR_W  = 16, 70

def draw_arrow_img(img, x1, y, x2):
    d = ImageDraw.Draw(img)
    d.line([(x1, y), (x2, y)], fill=(40, 40, 40), width=3)
    for dx, dy in [(-10, -6), (-10, 6)]:
        d.line([(x2, y), (x2 + dx, y + dy)], fill=(40, 40, 40), width=3)

prod_mol = Chem.MolFromSmiles(PRODUCT_SMILES)

for _, row in df_valid.iterrows():
    b     = int(row["beam"])
    ll    = float(row["log_likelihood"])
    r1    = Chem.MolFromSmiles(str(row["reactant_1"]).strip())
    r2    = Chem.MolFromSmiles(str(row["reactant_2"]).strip()) if str(row["reactant_2"]).strip() else None
    n_r   = 2 if r2 else 1
    pw    = n_r * RXNW + (30 if r2 else 0) + PAD * 2 + ARR_W + RXNW
    canvas = Image.new("RGB", (pw, RXNH + 46), (250, 252, 255))
    dc     = ImageDraw.Draw(canvas)
    dc.text((10, 6), f"Beam {b}   log-likelihood = {ll:.4f}", font=font_md, fill=(30, 50, 120))
    cx = PAD; ty = 42
    canvas.paste(smiles_to_png(r1, RXNW, RXNH, legend=str(row["reactant_1"])[:28]), (cx, ty))
    cx += RXNW
    if r2:
        dc.text((cx + 7, ty + RXNH // 2 - 8), "+", font=font_md, fill=(80, 80, 80))
        cx += 30
        canvas.paste(smiles_to_png(r2, RXNW, RXNH, legend=str(row["reactant_2"])[:28]), (cx, ty))
        cx += RXNW
    draw_arrow_img(canvas, cx + PAD, ty + RXNH // 2, cx + PAD + ARR_W - 16)
    cx += PAD + ARR_W
    canvas.paste(smiles_to_png(prod_mol, RXNW, RXNH, legend="Indole"), (cx, ty))
    path = os.path.join(IMG_DIR, f"indole_beam{b}_pathway.png")
    canvas.save(path)
    print(f"  ✓ Beam {b} 路径图: {path}")

print()

# ══════════════════════════════════════════════════════════════════
# 步骤 5：交互式 3D HTML 查看器
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 5：生成交互式 3D HTML 查看器")
print("=" * 60)

js_mols = []
for canon, info in mol_registry.items():
    opt_mol, sdf_block = optimized_mols[canon]
    pr = next((r for r in props_rows if r["smiles"] == canon), {})
    js_mols.append({
        "name":     info["name"],
        "smiles":   Chem.MolToSmiles(Chem.RemoveHs(opt_mol)),
        "sdf":      sdf_block,
        "mw":       pr.get("MW", ""),
        "logp":     pr.get("logP", ""),
        "hbd":      pr.get("HBD", ""),
        "hba":      pr.get("HBA", ""),
        "tpsa":     pr.get("TPSA", ""),
        "rotb":     pr.get("RotBonds", ""),
        "arom":     pr.get("AromaticRings", ""),
        "energy":   pr.get("MMFF94_energy_kcal", ""),
        "isTarget": canon == target_canon,
    })

js_data = json.dumps(js_mols, ensure_ascii=False, indent=2)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>吲哚逆合成 — 3D 分子查看器</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #eef2f7; color: #2c3e50; }}
header {{ background: linear-gradient(135deg, #1b4332, #2d6a4f); color: white;
          padding: 18px 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
header h1 {{ font-size: 1.45rem; }}
header p  {{ font-size: 0.82rem; opacity: 0.85; margin-top: 5px; line-height: 1.6; }}
#info-box {{ background: #d8f3dc; border-left: 4px solid #2d6a4f;
             padding: 10px 20px; margin: 14px 20px; border-radius: 4px;
             font-size: 13px; line-height: 1.8; }}
#controls {{ padding: 8px 22px; display: flex; align-items: center; gap: 18px;
             background: white; border-bottom: 1px solid #cde; font-size: 13px; }}
select {{ padding: 4px 8px; border: 1px solid #bbb; border-radius: 4px; font-size: 13px; }}
#grid {{ display: flex; flex-wrap: wrap; gap: 18px; padding: 20px; }}
.card {{ background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.12);
         overflow: hidden; width: 310px; transition: transform .15s, box-shadow .15s; }}
.card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.18); }}
.card.target {{ border: 3px solid #2d6a4f; }}
.card-header {{ padding: 9px 13px; background: #f0f7f4;
                border-bottom: 1px solid #cde; font-weight: 700; font-size: .88rem; }}
.card.target .card-header {{ background: #d8f3dc; color: #1b4332; }}
.viewer {{ width: 310px; height: 240px; position: relative; }}
.smiles {{ font-size: 9.5px; color: #888; padding: 4px 12px; word-break: break-all;
           background: #fafafa; border-bottom: 1px solid #eee; }}
.props {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px 4px;
          padding: 7px 12px 8px; font-size: 11.5px; }}
.prop {{ display: flex; justify-content: space-between; padding: 1px 0; }}
.prop span:first-child {{ color: #7f8c8d; }}
.prop span:last-child  {{ font-weight: 600; color: #1b2631; }}
.energy {{ font-size: 10.5px; text-align: center; padding: 4px 12px 7px;
           color: #5d6d7e; border-top: 1px solid #eee; }}
.tag {{ display: inline-block; font-size: 10px; background: #2d6a4f; color: white;
        padding: 1px 6px; border-radius: 10px; margin-left: 6px; }}
</style>
</head>
<body>
<header>
  <h1>吲哚 (Indole) — 逆合成 3D 分子查看器</h1>
  <p>目标分子: <code>c1ccc2[nH]ccc2c1</code> &nbsp;|&nbsp;
     Beam 数: {N_BEAMS} &nbsp;|&nbsp; 结构优化: MMFF94 ({N_CONFS} 构象搜索)<br>
     鼠标左键旋转 &nbsp;/&nbsp; 右键平移 &nbsp;/&nbsp; 滚轮缩放 &nbsp;|&nbsp; 需联网加载 3Dmol.js CDN</p>
</header>
<div id="info-box">
  <b>吲哚已知合成路线参考：</b><br>
  Fischer 合成：苯肼 + 醛/酮 &nbsp;|&nbsp;
  Larock 合成：邻卤苯胺 + 炔烃 &nbsp;|&nbsp;
  Leimgruber-Batcho：2-硝基甲苯 &nbsp;|&nbsp;
  Reissert 合成：邻硝基甲苯 → 靛红酸酐
</div>
<div id="controls">
  <span>显示样式：</span>
  <select id="styleSelect" onchange="changeStyle(this.value)">
    <option value="stick">棍棒 (Stick)</option>
    <option value="sphere">球 (Sphere)</option>
    <option value="line">线框 (Line)</option>
    <option value="cross">叉号 (Cross)</option>
  </select>
</div>
<div id="grid"></div>
<script>
const mols = {js_data};
const viewers = [];

function makeStyle(type) {{
  if (type === 'stick')  return {{stick: {{radius: 0.14}}, sphere: {{scale: 0.27}}}};
  if (type === 'sphere') return {{sphere: {{scale: 0.45}}}};
  if (type === 'line')   return {{line: {{}}}};
  if (type === 'cross')  return {{cross: {{lineWidth: 2}}}};
  return {{stick: {{radius: 0.14}}, sphere: {{scale: 0.27}}}};
}}

function changeStyle(type) {{
  viewers.forEach(v => {{ v.setStyle({{}}, makeStyle(type)); v.render(); }});
}}

const grid = document.getElementById('grid');
mols.forEach((mol, i) => {{
  const card = document.createElement('div');
  card.className = 'card' + (mol.isTarget ? ' target' : '');
  card.innerHTML = `
    <div class="card-header">${{mol.name}}${{mol.isTarget ? '<span class="tag">目标</span>' : ''}}</div>
    <div class="viewer" id="v${{i}}"></div>
    <div class="smiles">${{mol.smiles}}</div>
    <div class="props">
      <div class="prop"><span>MW</span><span>${{mol.mw}} g/mol</span></div>
      <div class="prop"><span>logP</span><span>${{mol.logp}}</span></div>
      <div class="prop"><span>HBD</span><span>${{mol.hbd}}</span></div>
      <div class="prop"><span>HBA</span><span>${{mol.hba}}</span></div>
      <div class="prop"><span>TPSA</span><span>${{mol.tpsa}} Å²</span></div>
      <div class="prop"><span>芳香环</span><span>${{mol.arom}}</span></div>
    </div>
    <div class="energy">MMFF94 最低能量: ${{mol.energy}} kcal/mol</div>
  `;
  grid.appendChild(card);
  const viewer = $3Dmol.createViewer(`v${{i}}`, {{backgroundColor: 'white', antialias: true}});
  viewer.addModel(mol.sdf, 'sdf');
  viewer.setStyle({{}}, makeStyle('stick'));
  viewer.addSurface($3Dmol.SurfaceType.VDW, {{
    opacity: 0.07,
    colorscheme: mol.isTarget ? 'greenCarbon' : 'grayCarbon'
  }});
  viewer.zoomTo();
  viewer.render();
  viewers.push(viewer);
}});
</script>
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as fh:
    fh.write(html)
print(f"  ✓ 3D 查看器: {HTML_PATH}\n")

# ══════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("全部完成！输出文件汇总（所有文件以 indole_ 开头）：")
print("=" * 60)
print(f"  逆合成结果:    {INDOLE_CSV}")
print(f"  分子性质表:    {PROP_CSV}")
print(f"  优化结构:      {OPT_DIR}/")
print(f"    indole_*.sdf / .xyz / .pdb")
print(f"  2D 图像:       {IMG_DIR}/")
print(f"    indole_molecules_grid.png")
print(f"    indole_beam*_pathway.png")
print(f"  3D HTML:       {HTML_PATH}")
print()
print("分子性质预览：")
print(props_df[["name","MW","logP","HBD","HBA","TPSA","AromaticRings","MMFF94_energy_kcal"]].to_string(index=False))
