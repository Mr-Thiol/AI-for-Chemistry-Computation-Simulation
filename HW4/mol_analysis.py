"""
mol_analysis.py
逆合成分子分析工具：结构优化 + 格式转换 + 2D/3D 可视化

功能：
1. 多构象搜索 + MMFF94/UFF 结构优化，报告收敛能量
2. 输出 SDF / XYZ / PDB 三种格式
3. 2D 结构网格图（PNG）+ 反应路径示意图（每条 beam）
4. 交互式 3D HTML 查看器（3Dmol.js，需联网加载 CDN）

用法：
    conda activate chemformer
    cd E:\\JupyterPjs\\AI4Chem\\HW4
    python mol_analysis.py
"""

import os
import json
import math
from io import BytesIO

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from PIL import Image, ImageDraw, ImageFont

# ── 路径配置 ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(SCRIPT_DIR, "retrosynthesis_result.csv")
OUT_DIR      = os.path.join(SCRIPT_DIR, "mol_analysis")
OPT_DIR      = os.path.join(OUT_DIR, "optimized")
IMG_DIR      = os.path.join(OUT_DIR, "images")
HTML_PATH    = os.path.join(OUT_DIR, "viewer_3d.html")
PROP_CSV     = os.path.join(OUT_DIR, "mol_properties.csv")

for d in [OUT_DIR, OPT_DIR, IMG_DIR]:
    os.makedirs(d, exist_ok=True)

PRODUCT_SMILES = "CCN(CC)CCNC(=O)C1=CC(=C(C=C1OC)N)Cl"
PRODUCT_NAME   = "Metoclopramide (Target)"
N_CONFS        = 50   # 构象搜索数量（增大可提升采样质量，但更慢）


# ══════════════════════════════════════════════════════════════════
# 1. 加载数据，收集所有唯一分子
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 1：加载逆合成数据")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

# 收集所有唯一分子：{canon_smiles: (display_name, beam_info)}
mol_registry = {}  # canon_smiles → {"name": str, "mol": Mol, "beams": [int]}

def register(smiles, name):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        print(f"  ✗ 无效 SMILES，跳过: {smiles}")
        return None
    canon = Chem.MolToSmiles(m)
    if canon not in mol_registry:
        mol_registry[canon] = {"name": name, "mol": m, "beams": []}
    return canon

target_canon = register(PRODUCT_SMILES, PRODUCT_NAME)
for _, row in df.iterrows():
    b = int(row["beam"])
    register(str(row["reactant_1"]).strip(), f"Beam {b} / 反应物 1")
    if str(row["reactant_2"]).strip():
        register(str(row["reactant_2"]).strip(), f"Beam {b} / 反应物 2")
    # 标注参与的 beam
    for canon, info in mol_registry.items():
        if canon in [Chem.MolToSmiles(Chem.MolFromSmiles(str(row["reactant_1"]).strip())),
                     Chem.MolToSmiles(Chem.MolFromSmiles(str(row["reactant_2"]).strip()))
                     if str(row["reactant_2"]).strip() else ""]:
            if b not in info["beams"]:
                info["beams"].append(b)

print(f"共收集到 {len(mol_registry)} 个唯一分子（含目标分子）\n")


# ══════════════════════════════════════════════════════════════════
# 2. 3D 结构优化
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 2：3D 结构优化（多构象 + MMFF94）")
print("=" * 60)

def optimize_3d(mol, name, n_confs=N_CONFS):
    """多构象搜索 + MMFF94 优化，返回 (opt_mol, energy_kcal, n_success)"""
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed   = 42
    params.numThreads   = 0   # 使用所有 CPU 核
    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params)
    n_success = len(conf_ids)
    if n_success == 0:
        print(f"  ✗ {name}: 构象嵌入失败，降级到 UFF")
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=2000)
        return mol_h, None, 0

    # 对每个构象做 MMFF94 优化，记录能量
    energies = []
    ff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
    for cid in conf_ids:
        if ff_props:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, ff_props, confId=cid)
            if ff:
                ff.Minimize(maxIts=2000)
                energies.append((ff.CalcEnergy(), cid))
            else:
                energies.append((float("inf"), cid))
        else:
            # fallback UFF
            AllChem.UFFOptimizeMolecule(mol_h, confId=cid, maxIters=2000)
            uff_ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
            e = uff_ff.CalcEnergy() if uff_ff else float("inf")
            energies.append((e, cid))

    energies.sort(key=lambda x: x[0])
    best_energy, best_cid = energies[0]

    # 构建只含最优构象的分子
    best_mol = Chem.RWMol(mol_h)
    best_mol = best_mol.GetMol()
    confs_to_remove = [cid for _, cid in energies[1:]]
    for cid in confs_to_remove:
        try:
            best_mol.RemoveConformer(cid)
        except Exception:
            pass

    return best_mol, best_energy, n_success

# 优化每个分子并保存 SDF / XYZ / PDB
props_rows = []
optimized_mols = {}   # canon_smiles → (opt_mol_with_H, sdf_block)

for canon, info in mol_registry.items():
    mol   = info["mol"]
    name  = info["name"]
    safe  = name.replace("/", "-").replace(" ", "_")
    print(f"  优化: {name}  SMILES={canon}")

    opt_mol, energy, n_ok = optimize_3d(mol, name)
    optimized_mols[canon] = opt_mol

    # ── 计算分子性质（使用无 H 的 2D mol）──────────────────────────
    mw    = Descriptors.MolWt(mol)
    logp  = Descriptors.MolLogP(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    tpsa  = Descriptors.TPSA(mol)
    rotb  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)

    props_rows.append({
        "name": name, "smiles": canon,
        "MW": round(mw, 2), "logP": round(logp, 2),
        "HBD": hbd, "HBA": hba, "TPSA": round(tpsa, 1),
        "RotBonds": rotb, "Rings": rings,
        "n_confs_tried": N_CONFS, "n_confs_ok": n_ok,
        "MMFF94_energy_kcal": round(energy, 3) if energy is not None else "N/A",
    })
    energy_str = f"{energy:.2f} kcal/mol" if energy is not None else "N/A"
    print(f"    ✓ 构象: {n_ok}/{N_CONFS} 成功  最低能量: {energy_str}")

    # ── SDF ─────────────────────────────────────────────────────
    sdf_path = os.path.join(OPT_DIR, f"{safe}.sdf")
    writer = Chem.SDWriter(sdf_path)
    opt_mol.SetProp("_Name", name)
    opt_mol.SetProp("SMILES", canon)
    if energy is not None:
        opt_mol.SetProp("MMFF94_Energy_kcal", str(round(energy, 4)))
    writer.write(opt_mol)
    writer.close()

    # ── XYZ ─────────────────────────────────────────────────────
    xyz_path = os.path.join(OPT_DIR, f"{safe}.xyz")
    conf = opt_mol.GetConformer()
    lines = [str(opt_mol.GetNumAtoms()), f"{name}  E={energy_str}"]
    for atom in opt_mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<3s}  {p.x:12.6f}  {p.y:12.6f}  {p.z:12.6f}")
    with open(xyz_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ── PDB ─────────────────────────────────────────────────────
    pdb_path = os.path.join(OPT_DIR, f"{safe}.pdb")
    Chem.MolToPDBFile(opt_mol, pdb_path)

    optimized_mols[canon] = (opt_mol, Chem.MolToMolBlock(opt_mol))

# 保存性质 CSV
props_df = pd.DataFrame(props_rows)
props_df.to_csv(PROP_CSV, index=False)
print(f"\n分子性质已保存: {PROP_CSV}\n")


# ══════════════════════════════════════════════════════════════════
# 3. 2D 结构网格图
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 3：生成 2D 结构网格图")
print("=" * 60)

MOL_W, MOL_H = 360, 280   # 单个分子图尺寸
FONT_H        = 22          # 图例行高
COLS          = 3
BG            = (255, 255, 255)
BORDER        = (220, 220, 220)

def smiles_to_png(mol, width=MOL_W, height=MOL_H, legend=""):
    """RDKit Cairo → PIL Image（带标注）"""
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.padding = 0.12
    drawer.DrawMolecule(mol, legend=legend)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))

# 目标分子 + 所有反应物
entries = [(target_canon, PRODUCT_NAME)] + [
    (c, i["name"]) for c, i in mol_registry.items() if c != target_canon
]

n_mols = len(entries)
rows   = math.ceil(n_mols / COLS)
grid_w = COLS * MOL_W
grid_h = rows * (MOL_H + FONT_H)
grid   = Image.new("RGB", (grid_w, grid_h), BG)
draw_ctx = ImageDraw.Draw(grid)

try:
    font = ImageFont.truetype("arial.ttf", 13)
except Exception:
    font = ImageFont.load_default()

for idx, (canon, name) in enumerate(entries):
    col = idx % COLS
    row = idx // COLS
    x   = col * MOL_W
    y   = row * (MOL_H + FONT_H)
    mol = mol_registry[canon]["mol"]
    # 简短标注：名称 + MW
    mw  = next(r["MW"] for r in props_rows if r["smiles"] == canon)
    img = smiles_to_png(mol, legend=f"MW={mw}")
    grid.paste(img, (x, y))
    # 底部文字标签
    draw_ctx.rectangle([x, y + MOL_H, x + MOL_W, y + MOL_H + FONT_H], fill=(245, 245, 250))
    draw_ctx.text((x + 6, y + MOL_H + 4), name[:38], font=font, fill=(50, 50, 80))
    # 边框
    draw_ctx.rectangle([x, y, x + MOL_W - 1, y + MOL_H + FONT_H - 1], outline=BORDER)

grid_path = os.path.join(IMG_DIR, "all_molecules_grid.png")
grid.save(grid_path)
print(f"  ✓ 网格图: {grid_path}\n")


# ══════════════════════════════════════════════════════════════════
# 4. 反应路径示意图（每条 beam）
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 4：生成反应路径示意图")
print("=" * 60)

RXNW, RXNH = 280, 220
PAD, ARR_W = 20, 80   # 箭头区域宽度

def draw_arrow(img, x1, y1, x2, y2, color=(50, 50, 50)):
    """PIL 上绘制带箭头的线"""
    d = ImageDraw.Draw(img)
    d.line([(x1, y1), (x2, y2)], fill=color, width=3)
    # 箭头头部
    for dx, dy in [(-10, -6), (-10, 6)]:
        d.line([(x2, y2), (x2 + dx, y2 + dy)], fill=color, width=3)

for _, row in df.iterrows():
    b       = int(row["beam"])
    ll      = float(row["log_likelihood"])
    r1_smi  = str(row["reactant_1"]).strip()
    r2_smi  = str(row["reactant_2"]).strip()
    r1_mol  = Chem.MolFromSmiles(r1_smi)
    r2_mol  = Chem.MolFromSmiles(r2_smi) if r2_smi else None

    # 布局：[R1] [+] [R2]  →→  [Product]
    n_r = 2 if r2_mol else 1
    plus_w = 30 if r2_mol else 0
    total_w = n_r * RXNW + plus_w + PAD * 2 + ARR_W + RXNW
    canvas  = Image.new("RGB", (total_w, RXNH + 50), (250, 252, 255))
    d_ctx   = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        title_font = small_font = ImageFont.load_default()

    # 标题
    d_ctx.text((10, 6), f"Beam {b}   log-likelihood = {ll:.4f}", font=title_font, fill=(40, 40, 100))

    cursor_x = PAD
    top_y    = 36

    # 反应物 1
    img_r1 = smiles_to_png(r1_mol, RXNW, RXNH, legend=r1_smi[:30])
    canvas.paste(img_r1, (cursor_x, top_y))
    cursor_x += RXNW

    # 加号
    if r2_mol:
        d_ctx.text((cursor_x + 7, top_y + RXNH // 2 - 8), "+", font=title_font, fill=(80, 80, 80))
        cursor_x += plus_w
        img_r2 = smiles_to_png(r2_mol, RXNW, RXNH, legend=r2_smi[:30])
        canvas.paste(img_r2, (cursor_x, top_y))
        cursor_x += RXNW

    # 箭头
    ax1 = cursor_x + PAD
    ax2 = cursor_x + PAD + ARR_W - 20
    ay  = top_y + RXNH // 2
    draw_arrow(canvas, ax1, ay, ax2, ay)
    cursor_x += PAD + ARR_W

    # 产品
    prod_mol = Chem.MolFromSmiles(PRODUCT_SMILES)
    img_prod = smiles_to_png(prod_mol, RXNW, RXNH, legend="Metoclopramide")
    canvas.paste(img_prod, (cursor_x, top_y))

    path = os.path.join(IMG_DIR, f"beam{b}_pathway.png")
    canvas.save(path)
    print(f"  ✓ Beam {b}: {path}")

print()


# ══════════════════════════════════════════════════════════════════
# 5. 交互式 3D HTML 查看器（3Dmol.js）
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("步骤 5：生成交互式 3D HTML")
print("=" * 60)

# 将分子数据序列化为 JavaScript 对象
js_mols = []
for canon, info in mol_registry.items():
    opt_mol, sdf_block = optimized_mols[canon]
    # 计算性质
    prop_row = next((r for r in props_rows if r["smiles"] == canon), {})
    # 读取 SMILES（无H）
    mol_noH = Chem.RemoveHs(opt_mol)
    smiles_clean = Chem.MolToSmiles(mol_noH)
    js_mols.append({
        "name":    info["name"],
        "smiles":  smiles_clean,
        "sdf":     sdf_block,
        "mw":      prop_row.get("MW", ""),
        "logp":    prop_row.get("logP", ""),
        "hbd":     prop_row.get("HBD", ""),
        "hba":     prop_row.get("HBA", ""),
        "tpsa":    prop_row.get("TPSA", ""),
        "rotb":    prop_row.get("RotBonds", ""),
        "energy":  prop_row.get("MMFF94_energy_kcal", ""),
        "isTarget": canon == target_canon,
    })

js_data = json.dumps(js_mols, ensure_ascii=False, indent=2)

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>甲氧氯普胺逆合成 — 3D 分子查看器</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #eef0f5; color: #333; }}
  header {{ background: linear-gradient(135deg, #1a5276, #2e86c1); color: white;
            padding: 18px 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
  header h1 {{ font-size: 1.5rem; }}
  header p  {{ font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }}
  #grid {{ display: flex; flex-wrap: wrap; gap: 18px; padding: 22px; }}
  .card {{ background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.12);
           overflow: hidden; width: 320px; transition: transform .15s, box-shadow .15s; }}
  .card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.18); }}
  .card.target {{ border: 2.5px solid #2e86c1; }}
  .card-header {{ padding: 10px 14px; background: #f4f6fb;
                  border-bottom: 1px solid #dde; font-weight: 600; font-size: .9rem; }}
  .card.target .card-header {{ background: #d6eaf8; color: #1a5276; }}
  .viewer {{ width: 320px; height: 250px; position: relative; }}
  .smiles {{ font-size: 10px; color: #888; padding: 4px 14px; word-break: break-all;
             background: #fafafa; border-bottom: 1px solid #eee; }}
  .props {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3px 6px;
            padding: 8px 14px 10px; font-size: 11.5px; }}
  .prop {{ display: flex; justify-content: space-between; padding: 1px 0; }}
  .prop span:first-child {{ color: #777; }}
  .prop span:last-child  {{ font-weight: 600; color: #2c3e50; }}
  .energy {{ font-size: 11px; text-align: center; padding: 4px 14px 8px;
             color: #5d6d7e; border-top: 1px solid #eee; }}
  #controls {{ padding: 8px 22px; display: flex; align-items: center; gap: 16px;
               background: white; border-bottom: 1px solid #dde; font-size: 13px; }}
  select {{ padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>
<header>
  <h1>甲氧氯普胺 (Metoclopramide) — 逆合成 3D 分子查看器</h1>
  <p>SMILES: CCN(CC)CCNC(=O)C1=CC(=C(C=C1OC)N)Cl &nbsp;|&nbsp;
     结构优化: MMFF94 ({N_CONFS} 构象搜索) &nbsp;|&nbsp;
     鼠标左键旋转 / 右键平移 / 滚轮缩放</p>
</header>
<div id="controls">
  <span>显示样式：</span>
  <select id="styleSelect" onchange="changeStyle(this.value)">
    <option value="stick">棍棒 (Stick)</option>
    <option value="sphere">球 (Sphere)</option>
    <option value="cartoon" disabled>卡通（蛋白专用）</option>
    <option value="line">线框 (Line)</option>
    <option value="cross">叉号 (Cross)</option>
  </select>
  <span style="margin-left:20px; color:#888; font-size:11px">需联网加载 3Dmol.js CDN</span>
</div>
<div id="grid"></div>

<script>
const mols = {js_data};

const viewers = [];

function makeStyle(type) {{
  if (type === 'stick')   return {{stick: {{radius: 0.14}}, sphere: {{scale: 0.28}}}};
  if (type === 'sphere')  return {{sphere: {{scale: 0.45}}}};
  if (type === 'line')    return {{line: {{}}}};
  if (type === 'cross')   return {{cross: {{lineWidth: 2}}}};
  return {{stick: {{radius: 0.14}}, sphere: {{scale: 0.28}}}};
}}

function changeStyle(type) {{
  viewers.forEach(v => {{
    v.setStyle({{}}, makeStyle(type));
    v.render();
  }});
}}

function atomColors(viewer) {{
  // 按元素上色（C=灰, N=蓝, O=红, Cl=绿, H=白）
  const colorMap = {{ C: '#888888', N: '#3355ff', O: '#ff2222',
                       Cl: '#1db954', H: '#eeeeee', S: '#e8c000' }};
  Object.entries(colorMap).forEach(([elem, color]) => {{
    viewer.setStyle({{elem}}, {{stick: {{colorscheme: 'elementColor', radius: 0.12}}}});
  }});
}}

const grid = document.getElementById('grid');

mols.forEach((mol, i) => {{
  // 卡片
  const card = document.createElement('div');
  card.className = 'card' + (mol.isTarget ? ' target' : '');

  card.innerHTML = `
    <div class="card-header">${{mol.name}}</div>
    <div class="viewer" id="v${{i}}"></div>
    <div class="smiles">${{mol.smiles}}</div>
    <div class="props">
      <div class="prop"><span>MW</span><span>${{mol.mw}} g/mol</span></div>
      <div class="prop"><span>logP</span><span>${{mol.logp}}</span></div>
      <div class="prop"><span>HBD</span><span>${{mol.hbd}}</span></div>
      <div class="prop"><span>HBA</span><span>${{mol.hba}}</span></div>
      <div class="prop"><span>TPSA</span><span>${{mol.tpsa}} Å²</span></div>
      <div class="prop"><span>RotBonds</span><span>${{mol.rotb}}</span></div>
    </div>
    <div class="energy">MMFF94 最低能量: ${{mol.energy}} kcal/mol</div>
  `;
  grid.appendChild(card);

  // 3Dmol 查看器
  const viewer = $3Dmol.createViewer(`v${{i}}`, {{
    backgroundColor: 'white',
    antialias: true
  }});
  viewer.addModel(mol.sdf, 'sdf');
  viewer.setStyle({{}}, makeStyle('stick'));
  viewer.addSurface($3Dmol.SurfaceType.VDW, {{
    opacity: 0.08,
    colorscheme: mol.isTarget ? 'blueCarbons' : 'grayCarbon'
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
    fh.write(html_content)
print(f"  ✓ 3D 查看器: {HTML_PATH}")
print("   （在浏览器中打开，需联网加载 3Dmol.js CDN）\n")


# ══════════════════════════════════════════════════════════════════
# 输出汇总
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("全部完成！输出文件汇总：")
print("=" * 60)
print(f"  分子性质表:    {PROP_CSV}")
print(f"  优化结构目录:  {OPT_DIR}/")
print(f"    - *.sdf  (含 MMFF94 优化结构与能量属性)")
print(f"    - *.xyz  (可用 VESTA / Avogadro 打开)")
print(f"    - *.pdb  (可用 PyMOL / Chimera 打开)")
print(f"  图像目录:      {IMG_DIR}/")
print(f"    - all_molecules_grid.png  (全部分子 2D 网格)")
print(f"    - beam*_pathway.png       (逐条反应路径)")
print(f"  3D HTML 查看器: {HTML_PATH}")
print()

# 打印性质表预览
print("分子性质预览：")
print(props_df[["name","MW","logP","HBD","HBA","TPSA","RotBonds","MMFF94_energy_kcal"]].to_string(index=False))
