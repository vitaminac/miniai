import matplotlib.pyplot as plt
import numpy as np

from .. import utils
from ..criterion import MSE
from ..opt import DifferentiableFunction
from ..train import GradientDescent


def h(x, parameters):
    return parameters['w'] * x + parameters['b']


class CostFunction(DifferentiableFunction):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.mse = MSE()

    def __call__(self, parameters):
        y_hat = h(self.x, parameters)
        return self.mse(self.y, y_hat)

    def grad(self, parameters):
        y_hat = h(self.x, parameters)
        # calculate the gradient
        delta = self.mse.grad(self.y, y_hat)
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
    model = CostFunction(x, y)
    gd = GradientDescent(model.grad, parameters, 1)
    losses = utils.collect_losses_and_log_to_tensorboard(
        gd.train(), model, parameters)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.show()
    print("Best parameters is ", parameters)


if __name__ == "__main__":
    show_demo()
