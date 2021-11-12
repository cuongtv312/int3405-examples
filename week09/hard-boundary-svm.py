import numpy as np
import math


def calculate(alpha, x, y):
    """
    Perform a check on the boundary create by alpha and calculate the boundary (objective function)
    :param alpha:
    :param x:
    :param y:
    :return:
    """
    if np.sum(alpha) > 0:
        w = np.sum(x[:, alpha == 1]*y[alpha == 1], axis=1).reshape((len(x), 1))

        b = np.sum(y[alpha == 1] - np.dot(w.T, x[:, alpha == 1]))
        b /= np.sum(alpha)

        # Perform check condition
        s = y*(np.dot(w.T, x) + b).reshape((-1, ))
        print(s.shape)
        if len(s[s > 0]) == len(s):
            return True, math.sqrt(np.sum(w*w))

    else:
        return False, -1e6


if __name__ == "__main__":
    X = [[1, 9],
         [5, 5],
         [1, 1],
         [8, 5],
         [12, 1],
         [10, 8]]
    y = [-1, -1, -1, 1, 1, 1]
    X = np.array(X).T
    y = np.array(y)

    mask = np.array([0, 1, 0, 1, 0, 0])
    flag, L = calculate(mask, X, y)
    print(flag, L)
