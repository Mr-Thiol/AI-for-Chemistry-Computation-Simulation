#!/usr/bin/env python3
"""
批量替换 LaTeX 文件中的化学式为规范格式
"""

import re

def fix_chemical_formulas(content):
    """修复化学式格式"""
    replacements = {
        # 下划线式改为规范式
        r'\\ce{CH_3SH}': r'\\ce{CH3SH}',
        r'\\ce{CH_3SCH_3}': r'\\ce{(CH3)2S}',
        r'\\ce{CH_3C\(=S\)CH_3}': r'\\ce{(CH3)2CS}',
        r'AllylSSAllyl': r'\\ce{(C3H5)2S2}',
        r'烯丙基二硫\\ \(AllylSSAllyl\)': r'二烯丙基二硫\\ (\\ce{(C3H5)2S2})',
        r'\\ce{N\(CH_3\)_3}': r'\\ce{(CH3)3N}',
        
        # 完整短语替换
        r'烯丙基二硫': r'二烯丙基二硫',
    }
    
    for old, new in replacements.items():
        content = re.sub(old, new, content)
    
    return content

# 读取文件
with open('report.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换
new_content = fix_chemical_formulas(content)

# 写回文件
with open('report.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("化学式已修复！")
