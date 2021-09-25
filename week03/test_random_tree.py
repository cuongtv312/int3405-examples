"""
Name:
Class:
MSSV:

You should understand your code

Several works can be added:
- Improve accuracy
- Improve speed: parrallel processing
- Add more parameters
"""

import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


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

class Node:

    def __init__(self, feature, value, left, right, depth, label):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label
        self.depth = depth


def calc_entropy(y):
    if len(y) <= 1:
        return 0

    n_label0 = len(y[y == 0])
    p = 1.*n_label0/len(y)
    if (p<1e-5) or abs(1-p) < 1e-5:
        return 0

    return -p*math.log(p) - (1-p)*math.log(1-p)


class RandomTree:

    def __init__(self, n_depth=6, n_random_features=0.1, seed=1):
        self.n_depth = n_depth
        self.n_random_features = n_random_features
        self.rng = np.random.RandomState(seed)

        self.tree = []

    def _fit(self, at, X, y):

        # If we reach n_depth
        if len(y) == 0:
            self.tree[at].label = 0

            return

        if self.tree[at].depth >= self.n_depth:
            n_label0 = len(y[y==0])
            n_label1 = len(y[y==1])
            if n_label0 < n_label1:
                self.tree[at].label = 1
            else:
                self.tree[at].label = 0

            return

        # we first calculate the best split
        best_ft = -1
        best_value = 1e9
        best_score = 1000
        for i in range(X.shape[1]):

            n_selected = int(len(X)*self.n_random_features)

            for j in range(n_selected):
                k = self.rng.randint(0, len(X))
                value = X[k, i] + 1e-5

                yleft = y[X[:, i] < value]
                yright = y[X[:, i] >= value]

                # You should change to gini score
                pleft = len(yleft) / len(y)
                pright = len(yright) / len(y)

                entropy = pleft*calc_entropy(yleft) + pright*calc_entropy(yright)

                if entropy < best_score:
                    best_score = entropy
                    best_ft = i
                    best_value = value

        self.tree[at].feature = best_ft
        self.tree[at].value = best_value
        self.tree[at].left = len(self.tree)
        self.tree[at].right = len(self.tree) + 1

        leftNode = Node(-1, -1, -1, -1, self.tree[at].depth+1, 1)
        rightNode = Node(-1, -1, -1, -1, self.tree[at].depth+1, 1)

        self.tree += [leftNode]
        self.tree += [rightNode]

        mask = X[:, best_ft] < best_value
        Xleft = X[mask]
        yleft = y[mask]

        mask = X[:, best_ft] >= best_value
        Xright = X[mask]
        yright = y[mask]

        self._fit(self.tree[at].left, Xleft, yleft)
        self._fit(self.tree[at].right, Xright, yright)

    def fit(self, X, y):
        node = Node(-1, -1, -1, -1, 0, 1)
        self.tree += [node]
        self._fit(0, X, y)

    def _predict(self, x):
        at = 0
        while True:
            i = self.tree[at].feature
            value = self.tree[at].value

            if i == -1:
                return self.tree[at].label

            if x[i] < value:
                at = self.tree[at].left
            else:
                at = self.tree[at].right

    def predict(self, X):
        ret = [self._predict(x) for x in X]

        return np.array(ret)


if __name__ == "__main__":
    np.random.seed(1)

    n_train = 10000
    std_train = 0.5

    n_test = 1000
    std_test = 0.5

    and_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_y = np.array([0, 0, 0, 1])

    Xtrain, ytrain = add_noise_data(and_X, and_y, n_train, 0., std_train)
    print(Xtrain.shape, ytrain.shape)

    rf = RandomTree()
    model = rf.fit(Xtrain, ytrain)

    Xtest, ytest = add_noise_data(and_X, and_y, n_test, 0., std_test)

    output_test = rf.predict(Xtest)
    print("Accuracy: ", accuracy_score(ytest, output_test))
    print("F1 score: ", f1_score(ytest, output_test))

