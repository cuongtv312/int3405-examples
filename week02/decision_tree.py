"""
Name:
Class:
MSSV:

You should understand your code
"""

import pandas as pd
import numpy as np
import math


def add_noise_data(input_data, input_labels, n_points, mean, scale):
    """
    Create a noise verstion of the input data

    Params:
        input_data: base input data
        input_labels: base input labels
        n_points: the number of needed points
        mean, scale: the gaussian data
    """
    raw_X = []
    raw_labels = []

    noise = np.random.normal(loc=mean, scale=scale, size=(n_points, 2))
    for i in range(n_points):
        k = np.random.randint(len(input_data))

        x1 = input_data[k][0] + noise[i][0]
        x2 = input_data[k][1] + noise[i][1]

        # We add more difficult for decision tree

        raw_X.append([x1 + x2, x1*x2,
                      math.sin(x1), 1/(1 + math.exp(-x2)), x1/abs(x2) + 1e-5])

        raw_labels.append(input_labels[k])

    return np.array(raw_X), np.array(raw_labels)


def build_decision_tree(X, y):
    return None


if __name__ == "__main__":
    np.random.seed(1)

    std = 0.2
    n_train = 10000
    n_test = 100

    and_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_y = np.array([0, 0, 0, 1])

    Xtrain, ytrain = add_noise_data(and_X, and_y, n_train, 0., std)

    model = build_decision_tree(Xtrain, ytrain)

    Xtest, ytest = add_noise_data(and_X, and_y, n_test, 0., std)
    print("Accuracy")
