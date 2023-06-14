import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from ..loss import Loss, MSELoss
from .. import utils
from ..train import GradientDescent


def h(x, parameters):
    return parameters['w'] * x + parameters['b']


class ModelWithManualGradient():
    def __init__(self, x, y,  loss: Loss):
        self.x = x
        self.y = y
        self.loss_function = loss

    def f(self, parameters):
        y_hat = h(self.x, parameters)
        return self.loss_function(self.y, y_hat)

    def f_grad(self, parameters):
        y_hat = h(self.x, parameters)
        # calculate the gradient
        delta = self.loss_function.derivative(self.y, y_hat)
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
    loss = MSELoss()
    model = ModelWithManualGradient(x, y, loss)
    gd = GradientDescent(model.f_grad, parameters, 1)
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
