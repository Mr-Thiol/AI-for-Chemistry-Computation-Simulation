"""
甲氧氯普胺逆合成分析 —— 本地 CPU 运行
使用 Chemformer (BART) backward_prediction 模式

运行方式：
    conda activate chemformer
    cd E:\JupyterPjs\AI4Chem\HW4
    python run_retrosynthesis.py
"""
import os
import sys

# 确保使用本地 molbart 源码
CHEMFORMER_DIR = os.path.join(os.path.dirname(__file__), "Chemformer")
sys.path.insert(0, CHEMFORMER_DIR)

# ── 配置 ───────────────────────────────────────────────────────────
PRODUCT_SMILES = "CCN(CC)CCNC(=O)C1=CC(=C(C=C1OC)N)Cl"   # 甲氧氯普胺
_MODEL_PATH_ORIG = os.path.join(os.path.dirname(__file__), "fine_tune_upsto_50_last.ckpt")
_MODEL_PATH_V2   = os.path.join(os.path.dirname(__file__), "fine_tune_upsto_50_last_v2.ckpt")
VOCAB_PATH       = os.path.join(CHEMFORMER_DIR, "bart_vocab_downstream.json")

# ── 修复 checkpoint：vocab_size → vocabulary_size ─────────────────
# （pytorch_lightning 的 load_from_checkpoint 依赖 hparams 中的键名）
import torch as _torch
if not os.path.exists(_MODEL_PATH_V2):
    print("修复 checkpoint（vocab_size → vocabulary_size）...")
    _ckpt = _torch.load(_MODEL_PATH_ORIG, map_location="cpu")
    _hp   = _ckpt.get("hyper_parameters", {})
    if "vocab_size" in _hp and "vocabulary_size" not in _hp:
        _hp["vocabulary_size"] = _hp.pop("vocab_size")
        _ckpt["hyper_parameters"] = _hp
        _torch.save(_ckpt, _MODEL_PATH_V2)
        print(f"已保存修复后的 checkpoint: {_MODEL_PATH_V2}")
    else:
        # 不需要修复，直接复用原文件路径
        _MODEL_PATH_V2 = _MODEL_PATH_ORIG
        print("checkpoint 键名已正常，无需修复。")
else:
    print("检测到已修复的 checkpoint，直接使用。")

MODEL_PATH = _MODEL_PATH_V2
N_BEAMS        = 5       # beam 数量（越大越慢，建议本地用 3~5）
OUTPUT_CSV     = os.path.join(os.path.dirname(__file__), "retrosynthesis_result.csv")

# ── 导入（顺序很重要，避免循环导入）─────────────────────────────────
print("导入模块...")
import molbart.utils.data_utils as util   # 先导入 data_utils 打破循环
from omegaconf import OmegaConf
from molbart.models import Chemformer
from molbart.data import SynthesisDataModule
import pandas as pd

# ── 加载模型 ───────────────────────────────────────────────────────
config = OmegaConf.create({
    "train_mode":            "eval",
    "batch_size":            1,
    "n_gpus":                0,          # CPU 模式
    "n_beams":               N_BEAMS,
    "n_unique_beams":        None,
    "vocabulary_path":       VOCAB_PATH,
    "model_path":            MODEL_PATH,
    "model_type":            "bart",
    "task":                  "backward_prediction",
    "data_path":             None,
    "dataset_part":          "full",
    "i_chunk":               0,
    "n_chunks":              1,
    "datamodule":            None,       # 使用 in-memory 模式
    "scorers":               None,
    "output_sampled_smiles": None,
})

print("加载模型（首次约需 10-30 秒）...")
chemformer = Chemformer(config)
print("模型加载完成！\n")

# ── 构建 in-memory 数据集并预测 ────────────────────────────────────
datamodule = SynthesisDataModule(
    reactants=[PRODUCT_SMILES],   # 占位，backward_prediction 时编码器输入是 products
    products=[PRODUCT_SMILES],
    tokenizer=chemformer.tokenizer,
    batch_size=1,
    max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
    dataset_path=""
)
datamodule.setup()

print(f"目标分子 (甲氧氯普胺): {PRODUCT_SMILES}")
print(f"正在预测（{N_BEAMS} beams，CPU 模式，请耐心等待）...\n")

import time
t0 = time.time()
smiles_beams, log_lhs_beams, original = chemformer.predict(
    dataloader=datamodule.full_dataloader()
)
elapsed = time.time() - t0

# ── 输出结果 ───────────────────────────────────────────────────────
print(f"预测完成！耗时 {elapsed:.1f} 秒\n")
print("=" * 60)
print(f"目标分子: {PRODUCT_SMILES}")
print("=" * 60)
print("逆合成预测结果（按概率排序）：")

rows = []
for i, (smi, lh) in enumerate(zip(smiles_beams[0], log_lhs_beams[0])):
    print(f"  Beam {i+1:2d}: {smi}")
    print(f"          log-likelihood = {lh:.4f}\n")
    parts = smi.split(".")
    rows.append({
        "beam":            i + 1,
        "reactants_smiles": smi,
        "log_likelihood":  lh,
        "n_reactants":     len(parts),
        "reactant_1":      parts[0] if len(parts) > 0 else "",
        "reactant_2":      parts[1] if len(parts) > 1 else "",
    })

# ── 保存 CSV ───────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"结果已保存至: {OUTPUT_CSV}")

# ── 生成 3D 结构（SDF + XYZ）─────────────────────────────────────
print("\n生成 3D 结构文件...")
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def smiles_to_3d(smiles, name="mol"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            return None
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        if props:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            if ff:
                ff.Minimize(maxIts=2000)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        return mol

    def save_xyz(mol, path, title=""):
        conf = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), title]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol():<3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out_dir = os.path.join(os.path.dirname(__file__), "molecules_3d")
    os.makedirs(out_dir, exist_ok=True)

    # 目标分子 + 所有 beam 中的独立分子
    to_process = {"target_metoclopramide": PRODUCT_SMILES}
    for row in rows:
        for j, part in enumerate(row["reactants_smiles"].split(".")):
            key = f"beam{row['beam']}_mol{j+1}"
            to_process[key] = part

    # 去重
    seen = {}
    for name, smi in to_process.items():
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else smi
        if canon not in seen:
            seen[canon] = name

    sdf_writer = Chem.SDWriter(os.path.join(out_dir, "all_molecules.sdf"))
    for canon_smi, name in seen.items():
        mol = smiles_to_3d(canon_smi, name)
        if mol is None:
            print(f"  ✗ 跳过: {name}")
            continue
        safe = name.replace(" ", "_")
        save_xyz(mol, os.path.join(out_dir, f"{safe}.xyz"), title=name)
        mol.SetProp("_Name", safe)
        mol.SetProp("SMILES", canon_smi)
        sdf_writer.write(mol)
        w = Chem.SDWriter(os.path.join(out_dir, f"{safe}.sdf"))
        w.write(mol); w.close()
        print(f"  ✓ {safe}.xyz / .sdf")
    sdf_writer.close()
    print(f"\n3D 文件保存至: {out_dir}")
    print("用 Avogadro 打开 all_molecules.sdf 可逐个浏览所有分子")

except Exception as e:
    print(f"3D 结构生成失败（不影响预测结果）: {e}")
