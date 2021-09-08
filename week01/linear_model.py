"""
A simple implementation of 2d Perceptron
"""

import numpy as np


class LinearModel2D:

    def __init__(self, a1, a2, b):
        """

        :param a1:
        :param a2:
        :param b:
        """
        self.a1 = a1
        self.a2 = a2
        self.b = b

    def __call__(self, x):

        # The index of numpy start from 1
        f = self.a1*x[0] + self.a2*x[1] + self.b

        return 1 if f >= 0.5 else 0

    def __str__(self):
        return f"a1 = {self.a1}, a2 = {self.a2}, b = {self.b}"


def get_accuracy(model, X, labels):

    count = 0
    for i in range(len(X)):
        if model(X[i]) == labels[i]:
            count += 1

    print("Accuracy = {}".format(1.0*count/len(X)))


# Main
if __name__ == "__main__":

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 0, 1]

    f1 = LinearModel2D(a1=1, a2=1, b=1)
    print(f1)
    get_accuracy(f1, X, labels)
