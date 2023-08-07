import numpy as np
import matplotlib.pyplot as plt
import torch
from ..criterion import MSE
from .. import utils
from ..train import GradientDescentWithAutoGrad
from ..train import BackpropagationWithAutoGrad


def h(x, params):
    return params['w'] * x + params['b']


def init_params():
    # https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
    w = torch.tensor(1, requires_grad=True, dtype=torch.float)
    b = torch.tensor(0, requires_grad=True, dtype=torch.float)
    return {'w': w, 'b': b}


def show_demo():
    utils.ensure_reproducity()
    # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    m = 1000
    x = torch.randn(m, dtype=torch.float)
    true_w = 197
    true_b = 769
    noise = 0.01 * torch.randn(m, dtype=torch.float)
    y = true_w * x + true_b + noise
    mse = MSE()

    def f(parameters):
        return mse(y, h(x, parameters))

    plt.figure(figsize=(10, 5))
    for learning_rate in np.logspace(-2, 0, 10):
        parameters = init_params()

        backprogation = BackpropagationWithAutoGrad(x, y, h, mse)
        gd = GradientDescentWithAutoGrad(
            backprogation, parameters, learning_rate)
        losses = utils.collect_losses(gd.train(), f, parameters)
        plt.plot(losses, label=str(learning_rate))
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend(loc='upper left')


if __name__ == "__main__":
    show_demo()
