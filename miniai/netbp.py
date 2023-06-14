import numpy as np
from .activation import sigmoid
from .loss import MSELoss
from .utils import show_data, show_costs, show_decision_boundary


def activate(x, W, b):
    return sigmoid(W @ x + b)


def h(W2, W3, W4, b2, b3, b4, X):
    return activate(activate(activate(X, W2, b2), W3, b3), W4, b4)


MSE_COST_FUNCTION = MSELoss()
def cost(W2, W3, W4, b2, b3, b4, X, Y):
    return MSE_COST_FUNCTION(Y, h(W2, W3, W4, b2, b3, b4, X))


# Uses backpropagation to train a network
def netbp():
    # Define Data
    X = np.array([
        [0.1, 0.1],
        [0.3, 0.4],
        [0.1, 0.5],
        [0.6, 0.9],
        [0.4, 0.2],
        [0.6, 0.3],
        [0.5, 0.6],
        [0.9, 0.2],
        [0.4, 0.4],
        [0.7, 0.6]
    ])
    Y = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]
    ])
    show_data(X, Y)
    # Initialize weights and biases
    np.random.seed(seed=5000)
    W2 = 0.5 * np.random.randn(2, 2)
    W3 = 0.5 * np.random.randn(3, 2)
    W4 = 0.5 * np.random.randn(2, 3)
    b2 = 0.5 * np.random.randn(2, 1)
    b3 = 0.5 * np.random.randn(3, 1)
    b4 = 0.5 * np.random.randn(2, 1)
    # Forward and Back propagate
    eta = 0.05  # learning rate
    Niter = int(1e6)  # number of SG iterations
    savecost = []  # value of cost function at each iteration
    for i in range(Niter):
        k = np.random.randint(10)  # choose a training point at random
        x = X[k, np.newaxis].T
        # Forward pass
        a2 = activate(x, W2, b2)
        a3 = activate(a2, W3, b3)
        a4 = activate(a3, W4, b4)
        # Backward pass
        delta4 = a4 * (1 - a4) * (a4 - Y[k, np.newaxis].T)
        delta3 = a3 * (1 - a3) * (W4.T @ delta4)
        delta2 = a2 * (1 - a2) * (W3.T @ delta3)
        # Gradient step
        W2 = W2 - eta*delta2*x.T
        W3 = W3 - eta*delta3*a2.T
        W4 = W4 - eta*delta4*a3.T
        b2 = b2 - eta*delta2
        b3 = b3 - eta*delta3
        b4 = b4 - eta*delta4
        # Monitor progress
        savecost.append(cost(W2, W3, W4, b2, b3, b4, X.T, Y.T))
    show_costs(savecost)
    show_decision_boundary(X, Y, lambda X: h(W2, W3, W4, b2, b3, b4, X.T))
