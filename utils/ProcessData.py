import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


def loadSpec(folder_name, range_=None, loadTime=None):
    entries = Path(folder_name)
    spectra_train = []
    irTime = []
    for entry in sorted(entries.iterdir()):

        if entry.name[0] == '.':
            continue
        print(entry.name)
        if loadTime:
            dateT = entry.name[loadTime[0]:loadTime[1]]
            hrs, mins, secs = float(dateT[:2]), float(dateT[3:5]), float(dateT[6:8])
            irTime.append(hrs * 60 * 60 + mins * 60 + secs)
        spectra_train.append(pd.read_csv(folder_name + entry.name).to_numpy())

    spectra_train = np.array(spectra_train)

    if range_:

        if len(range_) == 1:
            spectra_train = spectra_train[:, range_[0][0]:range_[0][1]]

        else:
            temp1 = spectra_train[:, range_[0][0]:range_[0][1]]
            for i in range(len(range_)-1):
                temp = spectra_train[:, range_[i+1][0]:range_[i+1][1]]
                temp1 = np.hstack((temp1, temp))

            spectra_train = temp1

    if loadTime:
        spectra_train = np.hstack((spectra_train[:, :, -1], np.array(irTime).reshape(-1, 1)))

    return spectra_train


def loadTemperature(filepath):
    temp = pd.read_excel(filepath).to_numpy()
    for i in range(temp.shape[0]):
        hrs, mins, secs = temp[i, 0].hour, temp[i, 0].minute, temp[i, 0].second
        temp[i, 0] = hrs * 60 * 60 + mins * 60 + secs
    return temp


def secondDeriv(data):

    temp = []
    for i in range(data.shape[0]):
        temp.append(-np.gradient(np.gradient(data[i, :])))
    temp = np.array(temp)

    return temp


def getErrorMatX(SVE, isPlot=False):

    x_mean = np.mean(SVE, axis=0)
    dSVE = SVE - x_mean
    xErrMat = (dSVE.T @ dSVE) / SVE.shape[0]

    if isPlot:
        na, nb = SVE.shape[0], SVE.shape[0]
        A = np.linspace(0, na, na)
        B = np.linspace(0, nb, nb)

        plt.figure(figsize=(12, 10))
        [AA, BB] = np.meshgrid(A, B)
        plt.contourf(AA, BB, xErrMat)
        plt.colorbar()
        plt.show()

    return xErrMat


# def plot3D(data):
#
#     df = pd.DataFrame(data=dataT[:, [0, 1, 2]], columns=['PC1', 'PC2', 'PC3'])
#     compound = 'time'
#
#     fig = plt.figure(figsize=(8, 6))
#     ax = Axes3D(fig, auto_add_to_figure=False)
#     fig.add_axes(ax)
#     x = df['PC1']
#     y = df['PC2']
#     z = df['PC3']
#     c = [i for i in range(dataT.shape[0])]
#     cb = ax.scatter(x, y, z, c=c, cmap='coolwarm')
#     index = 123
#     ax.scatter(x[index], y[index], z[index], c='r', marker='x')
#     ax.set_xlabel('pc1')
#     ax.set_ylabel('pc2')
#     ax.set_zlabel('pc3')
#     plt.colorbar(cb, shrink=0.5, label='time')
#     plt.show()