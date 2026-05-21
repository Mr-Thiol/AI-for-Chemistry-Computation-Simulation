import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

GAUSSIAN_SMEAR = 0.005
GRID_NUM = 200

# 二维高斯展宽函数 (用于在势能面上平滑打点)
def Gaussian2D(xx, yy, sigmax=0, sigmay=0, mux=0, muy=0):
    rtn = (xx - mux)**2 / sigmax**2 + (yy - muy)**2 / sigmay**2
    return np.exp(-0.5 * rtn) * 1 / (2 * np.pi * sigmax * sigmay)

# 核心画图函数
def PlotContour(l_val, suffix):
    infile = f"Q{l_val}E_{suffix}"
    outfile = f"QEPlot_Q{l_val}_{suffix}.png"
    xType = f"Q{l_val}"
    yType = "E"
    
    # 检查数据文件是否存在
    if not os.path.exists(infile):
        print(f"找不到数据文件 {infile}，跳过绘图...")
        return

    print(f"正在处理 {infile} 并绘制 {outfile}...")
    
    # 读取 LASP 生成的格式表格
    qeTable = pd.read_table(infile, sep=r"\s+", header=None, names=[xType, yType])
    
    startX = qeTable[xType].min()
    endX = qeTable[xType].max()
    startY = qeTable[yType].min()
    endY = qeTable[yType].max()
    
    rangeX = np.linspace(startX, endX, GRID_NUM)
    rangeY = np.linspace(startY, endY, GRID_NUM)
    X, Y = np.meshgrid(rangeX, rangeY)
    
    smearX = (endX - startX) * GAUSSIAN_SMEAR
    smearY = (endY - startY) * GAUSSIAN_SMEAR
    
    dataGrid = np.zeros((GRID_NUM, GRID_NUM))
    
    # 进行高斯展宽计算网格密度
    for idx, row in qeTable.iterrows():
        dataGrid += Gaussian2D(X, Y, smearX, smearY, row[xType], row[yType])
        
    # 开始绘图
    plt.figure(figsize=(8, 6))
    main_ax = plt.subplot()
    
    cntr = plt.contourf(
        X,
        Y,
        np.log(dataGrid + 1),  # 取对数让微小特征更明显
        cmap="jet",
    )
    
    # 设置坐标轴外观
    main_ax.xaxis.set_label_text(xType, size=18, color='black')
    main_ax.yaxis.set_label_text(yType, size=18, color='black')
    main_ax.tick_params(axis="both", top=False, bottom=True, left=True, right=True, labelsize=12, pad=0.5)
    
    plt.colorbar(cntr, label="log(Density + 1)")
    plt.title(f"Potential Energy Surface Contour ({xType} - {suffix.upper()})", size=14)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"成功保存图片: {outfile}")

if __name__ == "__main__":
    print("开始自动生成所有 Q-E 势能面等高线图...")
    # 自动遍历生成 L=2, 4, 6 以及 SSW/NVT 的 6 张图
    for l in [2, 4, 6]:
        for mode in ["ssw", "nvt"]:
            PlotContour(l, mode)
    print("太棒了！所有势能面绘图工作全部完成！")