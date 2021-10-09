"""
Name:
Class:
MSSV:

You should understand your code
"""

import pandas as pd
import numpy as np
import math
from sklearn.metrics import f1_score, accuracy_score


def add_noise_data_2input_1output(input_data, input_labels, n_points, mean, scale):
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

        raw_X.append([x1, x2])

        raw_labels.append(input_labels[k])

    return np.array(raw_X), np.array(raw_labels)


class SimpleMLP:

    def __init__(self, n_inputs=2, n_hidden=5, loss='MSE', learning_rate=1e-1, n_epochs=10):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.hidden_w = np.zeros_like([n_inputs + 1, n_hidden])
        self.output_w = np.zeros_like([n_hidden, 1])

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

        # stop training model after n_epoch epoch
        # you can add other stopping conditions, i.e loss on valid set doesn't decrease after some epochs.
        for i in range(self.n_epochs):
            self.update_parameters()
        return self

    def update_parameters(self):
        # TODO: compute gradients and then update parameters with gradient descent
        return self

    def predict(self, X):
        # TODO: update value return
        return [1 for _ in X]


if __name__ == "__main__":
    np.random.seed(1)

    std = 0.2
    n_train = 100
    n_test = 10

    and_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_y = np.array([0, 1, 1, 0])

    Xtrain, ytrain = add_noise_data_2input_1output(and_X, and_y, n_train, 0., std)
    print(Xtrain.shape, ytrain.shape)

    print('MSE results')
    mlp = SimpleMlp(2, 5, 'MSE')
    mlp.fit(Xtrain, ytrain)

    Xtest, ytest = add_noise_data_2input_1output(and_X, and_y, n_test, 0., std)
    print("Accuracy")

    output_test = mlp.predict(Xtest)
    print("Accuracy: ", accuracy_score(ytest, output_test))
    print("F1 score: ", f1_score(ytest, output_test))

    print('Cross Entropy results')
    mlp = SimpleMlp(2, 5, 'CE')
    mlp.fit(Xtrain, ytrain)

    Xtest, ytest = add_noise_data_2input_1output(and_X, and_y, n_test, 0., std)
    print("Accuracy")

    output_test = mlp.predict(Xtest)
    print("Accuracy: ", accuracy_score(ytest, output_test))
    print("F1 score: ", f1_score(ytest, output_test))
