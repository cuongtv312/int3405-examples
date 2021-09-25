"""
Implement a simple peceptron model

Name:
Class:
MSSV:

You should understand your code
"""

import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score


class Perceptron:

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        return None

    def predict(self, X):
        return [1 for _ in X]


def read_data(input_file):
    df = pd.read_csv(input_file)

    feature_columns = ['x1', 'x2']

    X = df[feature_columns].values
    y = df['y'].values

    return X, y


if __name__ == "__main__":
    np.random.seed(1)

    Xtrain, ytrain = read_data('./train_perceptron.csv')
    print(Xtrain.shape, ytrain.shape)

    model = Perceptron()
    model.fit(Xtrain, ytrain)

    print(model.w, model.b)
