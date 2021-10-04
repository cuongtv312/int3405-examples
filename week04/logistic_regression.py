"""
Implement a simple logistic regression model

Name:
Class:
MSSV:

You should understand your code
"""

import pandas as pd
import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate, n_epochs):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y_predict, y):
        # TODO: Compute loss function of model
        loss = None
        return loss

    # function for train model
    def fit(self, X, y):
        self.X = X
        self.y = y

        # TODO: Init w, b
        self.w = None
        self.b = None

        # stop training model after n_epoch epoch
        # you can add other stopping conditions, i.e loss on valid set doesn't decrease after some epochs.
        for i in range(self.n_epochs):
            self.update_parameters()
        return self

    def update_parameters(self):
        # TODO: compute gradients and then update parameters with gradient descent
        self.w = None
        self.b = None
        return self

    def predict(self, X):
        # TODO: update value return
        return [1 for _ in X]


def read_data(input_file):
    df = pd.read_csv(input_file)

    feature_columns = ['x1', 'x2']

    X = df[feature_columns].values
    y = df['y'].values

    return X, y



if __name__ == "__main__":
    np.random.seed(1)

    X_train, y_train = read_data('./train_perceptron.csv')
    print(X_train.shape, y_train.shape)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print(model.w, model.b)
