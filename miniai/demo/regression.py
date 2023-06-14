import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from ..loss import Loss, MSELoss
from .. import utils
from .gd import GradientDescent


def h(x, parameters):
    return parameters['w'] * x + parameters['b']


class GradientDescentWithManualGradient(GradientDescent):
    def __init__(self, x, y, initial_parameters, loss: Loss, learning_rate: float):
        super(GradientDescentWithManualGradient, self).__init__(
            x, y, initial_parameters, loss, learning_rate)

    def forward(self):
        self.y_hat = h(self.x, self.parameters)
        return self.y_hat

    def grad(self):
        # calculate the gradient
        delta = self.loss_function.derivative(self.y, self.y_hat)
        # calculate the gradient
        dw = (delta * self.x).mean()
        db = delta.mean()
        return {'w': dw, 'b': db}


def init_params():
    w = np.array([0.0])
    b = np.array([0.0])
    return {'w': w, 'b': b}


def show_demo():
    utils.ensure_reproducity()
    m = 1000
    x = np.random.randn(m)
    true_w = 197
    true_b = 769
    noise = 0.01 * np.random.randn(m)
    y = true_w * x + true_b + noise
    parameters = init_params()
    gd = GradientDescentWithManualGradient(x, y, parameters, MSELoss(), 1)
    losses = gd.train()
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.show()
    print("Best parameters is ", parameters)


if __name__ == "__main__":
    show_demo()
