import numpy.linalg as linalg
import numpy as np
import copy
from sklearn.preprocessing import normalize


class RobustLinear:

    def __init__(self, errorMat, lambda_, normalize=False):
        self.errorMat = copy.copy(errorMat)
        self.normalize = normalize
        self.lambda_ = lambda_

    def fit(self, x_, y_):

        x, y = copy.copy(x_).astype(float), copy.copy(y_).astype(float)
        x, y = self.preprocess(x, y)
        res = self.SolveSVE(x, y)
        self.w = res[1:]
        self.b = res[0]

    def predict(self, xtest_):

        xtest = copy.copy(xtest_).astype(float)
        if self.normalize:
            xtest = (xtest - self.xoffset) / self.xscale
        else:
            xtest = xtest - self.xoffset

        return xtest @ self.w + self.b + self.yoffset

    def SolveSVE(self, x, y):

        design_x = np.hstack((np.ones((x.shape[0], 1)), x))
        inv_x = np.linalg.inv(design_x.T @ design_x + self.lambda_ * self.errorMat)

        return (inv_x @ design_x.T @ y)

    def preprocess(self, x, y):

        self.xoffset = np.mean(x, axis=0)
        x -= self.xoffset
        if self.normalize:
            x, self.xscale = normalize(x, axis=0, copy=False, return_norm=True)

        self.yoffset = np.mean(y)
        y -= self.yoffset
        return x, y

