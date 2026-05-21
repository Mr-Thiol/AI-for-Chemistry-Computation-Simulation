import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

QE_IN_TABLE = "Q2E_ssw"
QE_OUT_FIGURE = "QEPlot.png"
GRID_DUMP = "grid.dump"
GAUSSIAN_SMEAR = 0.005
GRID_NUM = 200


def Gaussian2D(xx, yy, sigmax=0, sigmay=0, mux=0, muy=0):
    rtn = (xx - mux)**2 / sigmax**2 + (yy - muy)**2 / sigmay**2
    return np.exp(-0.5 * rtn) * 1 / 2 / np.pi / sigmax / sigmay


def CalCulateGrid(xType="Q2", yType="E"):
    qeTable = pd.read_table(QE_IN_TABLE,
                            sep=r"\s+",
                            header=None,
                            names=["Q2", "E"])
    dataGrid = np.zeros((GRID_NUM, GRID_NUM))

    startX = qeTable[xType].min()
    endX = qeTable[xType].max()
    startY = qeTable[yType].min()
    endY = qeTable[yType].max()
    print("xRange for data is: " + str([startX, endX]))
    print("yRange for data is: " + str([startY, endY]))

    smearX = (endX - startX) * GAUSSIAN_SMEAR
    smearY = (endY - startY) * GAUSSIAN_SMEAR
    print("xSmear, ySmear is: " + str((smearX, smearY)))

    nExcess = 10
    startX -= nExcess * smearX
    endX += nExcess * smearX
    print("xRange -+ %f" % (nExcess * smearX))
    startY -= nExcess * smearY
    endY += nExcess * smearY
    print("yRange -+ %f" % (nExcess * smearY))

    rangeX = np.linspace(startX, endX, GRID_NUM)
    rangeY = np.linspace(startY, endY, GRID_NUM)

    XX, YY = np.meshgrid(rangeX, rangeY)
    rowTotal = qeTable.shape[0]

    for i, row in qeTable.iterrows():
        dataGrid += Gaussian2D(XX, YY, smearX, smearY, row[xType], row[yType])
        if i % 1000 == 0:
            print("%d / %d" % (i, rowTotal))

    with open(GRID_DUMP, "wb") as fp:
        pickle.dump([rangeX, rangeY, dataGrid, xType, yType], fp)
    print("CalCulateGrid Done!")


def PlotFigure():
    with open(GRID_DUMP, "rb") as fp:
        rangeX, rangeY, dataGrid, xType, yType = pickle.load(fp)

    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.08,
                        bottom=0.08,
                        right=0.92,
                        top=0.92,
                        hspace=0.1,
                        wspace=0.1)
    main_ax = plt.subplot()

    X, Y = np.meshgrid(rangeX, rangeY)
    # levels = np.linspace(0, 14, 11)
    # levels[0] = 0.02
    # levels[-1] = 16
    levels = None
    # [
    #     cm.jet, cm.nipy_spectral, cm.CMRmap, cm.gnuplot2, cm.bwr, cm.seismic,
    #     cm.hot, cm.Greys, cm.rainbow
    # ]

    cntr = plt.contourf(
        X,
        Y,
        np.log(dataGrid + 1),
        # levels=levels,
        # alpha=0.8,
        cmap="jet",
    )
    print("levels: "+ str(cntr.levels))

    main_ax.xaxis.set_label_text(xType, size=18, color='black')
    main_ax.yaxis.set_label_text(yType, size=18, color='black')
    # main_ax.axis([-0.1, 2.1, -0.1, 3.4])

    main_ax.tick_params(axis="both",
                        top=False,
                        bottom=True,
                        left=True,
                        right=True,
                        labelsize=12,
                        pad=0.5)

    # cbar_sub = plt.subplot()
    # cbar_sub.axis("off")
    cbar = plt.colorbar(
        cntr,
        format='',
        # shrink=0.3,
        orientation='vertical',
        spacing='proportional')

    cbar.set_ticks([1, 6, 10])
    strList = ["Low", "DOS", "High"]
    cbar.ax.set_yticklabels(strList, fontsize=16, rotation=270)
    cbar.ax.tick_params(width=0)

    plt.savefig(QE_OUT_FIGURE, dpi=350)
    plt.show()


if __name__ == "__main__":

    CalCulateGrid(xType="Q2", yType="E")
    PlotFigure()
