import os
import glob
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# 1. 设定你的本地目标文件夹路径 (前面加 r 防止转义)
work_dir = r"E:\JupyterPjs\AI4Chem\HW5\pretrained\sample_2026_05_14__08_36_28_sample\xyzoutputs"

# 2. 构造搜索模式，找到目录下所有的 .xyz 文件
xyz_pattern = os.path.join(work_dir, "*.xyz")
xyz_files = glob.glob(xyz_pattern)

if not xyz_files:
    print(f"❌ 在 {work_dir} 目录下没有找到任何 .xyz 文件，请检查路径是否完全一致！")
else:
    print(f"🔍 共找到 {len(xyz_files)} 个 .xyz 文件，准备开始注入灵魂...\n")

# 3. 遍历并处理每个文件
for xyz_file in xyz_files:
    try:
        # 获取纯文件名（用于打印美观）
        base_name = os.path.basename(xyz_file)
        
        # 读取无键信息的 XYZ 坐标文件
        raw_mol = Chem.MolFromXYZFile(xyz_file)
        if raw_mol is None:
            print(f"⚠️ 警告：无法解析 {base_name}，可能格式不规范，跳过。")
            continue
            
        # 核心步骤：让 RDKit 根据 3D 距离推断单双键
        rdDetermineBonds.DetermineConnectivity(raw_mol)
        rdDetermineBonds.DetermineBondOrders(raw_mol, charge=0)
        
        # 补齐所有氢原子，并将其放置在符合 VSEPR 理论的合理三维空间位置
        mol_with_h = Chem.AddHs(raw_mol, addCoords=True)
        
        # 构造新的文件名并保存为 .mol 格式（强烈建议存为 mol，保留了键连关系）
        new_filename = xyz_file.replace(".xyz", "_Full_32Atoms.mol")
        Chem.MolToMolFile(mol_with_h, new_filename)
        
        print(f"✅ 成功补全: {base_name}")
        print(f"   -> 导出至: {os.path.basename(new_filename)} (当前原子总数: {mol_with_h.GetNumAtoms()})")
        
    except Exception as e:
        print(f"❌ 翻车了！处理 {base_name} 时发生错误: {e}")

print("\n🎉 全部执行完毕！快去那个文件夹里看看新鲜出炉的 32 原子全量分子吧！")