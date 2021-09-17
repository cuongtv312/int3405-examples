"""
Name:
Class:
MSSV:

You should understand your code

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
        self.w = np.array([a1, a2])
        self.b = b

    def __call__(self, x):

        # The index of numpy start from 1
        f = np.dot(x, self.w) + self.b*np.ones(len(x))

        return np.where(f >= 0.5, 1, 0)

    def __str__(self):
        return f"w = {self.w}, b = {self.b}"


# Main
if __name__ == "__main__":

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    accuracy = []

    np.random.seed(1)
    # Q1: Update to calculate the expected accuracy in case of N(0, 1)
    for _ in range(10000):
        a1 = np.random.uniform(-1, 1)
        a2 = np.random.uniform(-1, 1)
        f = LinearModel2D(a1, a2, 1)

        output = f(X)
        accuracy += [np.mean(np.where(output == y, 1, 0))]

    print("Empirical Accuracy = ", np.mean(accuracy))
