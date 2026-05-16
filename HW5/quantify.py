import os
import glob
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

# 1. 设定你的工作目录
work_dir = r"E:\JupyterPjs\AI4Chem\HW5\pretrained\sample_2026_05_14__08_36_28_sample\xyzoutputs"
xyz_files = glob.glob(os.path.join(work_dir, "*.xyz"))

print("==================================================")
print("🚀 AI4S 生成分子三维构象量化评估系统启动...")
print("==================================================\n")

# --- 准备“标准答案” (Ground Truth) ---
# 用 RDKit 传统的 ETKDG 算法生成一个完美的维生素 B7 作为基准
biotin_smiles = "O=C1N[C@@H]2SC[C@H](CCCCC(=O)O)[C@@H]2N1"
ref_mol = Chem.MolFromSmiles(biotin_smiles)
ref_mol_h = Chem.AddHs(ref_mol)
# 生成 3D 坐标的正确姿势
params = AllChem.ETKDGv3()
params.randomSeed = 42
AllChem.EmbedMolecule(ref_mol_h, params)
AllChem.MMFFOptimizeMolecule(ref_mol_h) # 力场极小化到最优态
ref_heavy = Chem.RemoveHs(ref_mol_h)    # 剥离氢，只留 16 个重原子用于对齐

success_count = 0

for xyz in xyz_files:
    base_name = os.path.basename(xyz)
    try:
        # 读取模型生成的 XYZ
        gen_mol = Chem.MolFromXYZFile(xyz)
        
        # 1. 尝试推断化学键 (拓扑有效性检查)
        rdDetermineBonds.DetermineConnectivity(gen_mol)
        rdDetermineBonds.DetermineBondOrders(gen_mol, charge=0)
        
        # 剥离模型随机生成的极个别氢原子，只保留核心重原子
        gen_heavy = Chem.RemoveHs(gen_mol)
        
        # --- 量化指标 1：力场能量差值 (Delta E) ---
        # 提取当前生成的能量
        ff_init = AllChem.MMFFGetMoleculeForceField(gen_heavy, AllChem.MMFFGetMoleculeProperties(gen_heavy))
        if ff_init is None:
            print(f"❌ [{base_name}] 评估失败: 拓扑严重错误，无法建立物理力场 (存在严重的原子穿透或断键)！")
            continue
        e_init = ff_init.CalcEnergy()
        
        # 模拟弛豫优化
        ff_opt = AllChem.MMFFGetMoleculeForceField(gen_heavy, AllChem.MMFFGetMoleculeProperties(gen_heavy))
        ff_opt.Minimize(maxIts=500)
        e_opt = ff_opt.CalcEnergy()
        
        delta_e = abs(e_init - e_opt)
        
        # --- 量化指标 2：重原子空间均方根误差 (Heavy-atom RMSD) ---
        # 寻找生成分子与标准基准分子的原子映射关系
        match = gen_heavy.GetSubstructMatch(ref_heavy)
        
        if match:
            # 对齐两个分子并计算 RMSD
            rmsd = AllChem.AlignMol(gen_heavy, ref_heavy, atomMap=list(enumerate(match)))
        else:
            rmsd = float('inf')
            
        # 打印量化报告
        print(f"✅ [{base_name}] 评估完成:")
        print(f"   -> 初始构象能量 : {e_init:8.2f} kcal/mol")
        print(f"   -> 优化后能量   : {e_opt:8.2f} kcal/mol")
        print(f"   -> 能量差值(ΔE): {delta_e:8.2f} kcal/mol")
        
        if rmsd != float('inf'):
            print(f"   -> 重原子 RMSD : {rmsd:8.3f} Å")
            success_count += 1
        else:
            print(f"   -> 重原子 RMSD : 计算失败 (生成的骨架图与真实 B7 不匹配)")
        print("-" * 50)
            
    except Exception as e:
        print(f"❌ [{base_name}] 评估出错: 结构高度畸变导致 RDKit 崩溃 ({e})")
        print("-" * 50)

print(f"\n📊 最终拓扑成功率 (Validity): {success_count} / {len(xyz_files)} ({(success_count/len(xyz_files))*100:.1f}%)")