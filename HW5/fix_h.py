from rdkit import Chem
import os

# 你的残缺版分子路径
old_mol_path = "/content/drive/MyDrive/Homework_AI4Chem/AI-for-Chemistry-Computation-Simulation/HW5/geodiff_modified/my_mols/Biotin.mol"

mol = Chem.MolFromMolFile(old_mol_path)

if mol is not None:
    # 核心魔法：补全所有的显式氢原子
    mol_with_H = Chem.AddHs(mol)
    
    # 覆盖保存
    Chem.MolToMolFile(mol_with_H, old_mol_path)
    
    print("✅ 补氢大获成功！")
    print(f"原来的原子数: {mol.GetNumAtoms()} (残缺版)")
    print(f"现在的原子数: {mol_with_H.GetNumAtoms()} (完美的 32 原子维生素 B7！)")
else:
    print("❌ 读取失败，请检查文件路径是否正确。")
