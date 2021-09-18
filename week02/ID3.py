"""
Name:
Class:
MSSV:

You should understand your code
"""

import pandas as pd
import numpy as np


# Q2.1
def compute_infomation_gain(X, y, feature):
    return 0


# Build decision tree on X and y
# List of:
# node_index, node_feature[0..3], (feature_value -> child_index) : internal node
# leafnode: node_index, node_features = -1, Yes/No
def build_ID3(X, y):
    return None


if __name__ == "__main__":
    df = pd.read_csv("./playtennis.csv")
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    target = ['PlayTennis']

    X = df[features].values
    y = df[target].values
    print("Input: ", X.shape, y.shape)

    # Q2.1
    print("Information Gain of Outlook: ", compute_infomation_gain(X, y, 0))

    # Q2.2
    build_ID3(X, y)
