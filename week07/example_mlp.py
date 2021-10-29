import numpy as np

np.random.seed(1)
W1 = np.random.rand(3, 2)
W2 = np.random.rand(3, 1)

print(W2)
print(W1.shape, W2.shape)

x1 = np.array([[1., 0., 1], [0., 1., 1]]).T
y1 = np.array([[1.], [1.]])
print(x1)
print(y1)

def _sigmoid(x):
    return 1./(1+np.exp(-x))

def forward(W1, W2, x):
    h = W1.T.dot(x)
    h = _sigmoid(h)

    o = W2.T.dot(np.concatenate([h, np.array([[1]])]))
    o = _sigmoid(o)

    return h, o

print("Forwarding")

lr = 10.
def train_1sample(W1, W2, x, t):
    h, y = forward(W1, W2, x1[:, 0].reshape(-1, 1))
    #print(h, y)

    E = 0.5*(t - y)**2

    dG = (t-y)*(y*(1-y))
    # w2 = lr*(y - o)*(o*(1-o))*W2
    dW2 = lr*dG*h
    dW2_b = lr*dG
    #print(dW2, dW2_b)

    """
    yD = h[0][0]
    dD = W2[0][0]*dG*yD*(1-yD)
    print(dD)

    yE = h[1][0]
    dE = W2[1][0]*dG*yE*(1-yE)

    print(dE)
    """

    deltaW1 = W2[:2] * dG * h*(1 - h)
    #print(deltaW1)

    dW1 = lr*deltaW1*x
    #print(dW1)

    return dW2, dW2_b, dW1


first_gradient = train_1sample(W1, W2, x1[:, 0], y1[0])
second_gradient = train_1sample(W1, W2, x1[:, 1], y1[1])

W1 += (first_gradient[2].T + second_gradient[2].T)/2
print(W1)