"""
Implement Naive Bayes model, using smoothing constant L

Name:
Class:
MSSV:

You should understand your code
"""

import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score


class NaiveBayes:

    def __init__(self, L=0):
        self.L = L

    def fit(self, X, y):
        return None

    def predict(self, X):
        return [1 for _ in X]

    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in X]


def read_data(input_file):
    df = pd.read_csv(input_file)

    feature_columns = [c for c in df.columns if c != 'PlayTennis']

    X = df[feature_columns].values

    # Convert to 0, 1
    _y = df['PlayTennis'].values
    y = np.where(_y == 'Yes', 1, 0)

    return X, y


if __name__ == "__main__":
    np.random.seed(1)

    Xtrain, ytrain = read_data('./train_nb.csv')
    Xtest, ytest = read_data('./test_nb.csv')
    print(Xtrain.shape, ytrain.shape)

    nb = NaiveBayes()
    nb.fit(Xtrain, ytrain)

    output_test = nb.predict(Xtest)
    print("Accuracy: ", accuracy_score(ytest, output_test))
    print("F1 score: ", f1_score(ytest, output_test))
