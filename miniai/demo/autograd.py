import numpy as np
import matplotlib.pyplot as plt
import torch
from ..loss import Loss, MSELoss
from .. import utils
from .gd import GradientDescent


def h(x, params):
    return params['w'] * x + params['b']


def init_params():
    # https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
    w = torch.tensor(1, requires_grad=True, dtype=torch.float)
    b = torch.tensor(0, requires_grad=True, dtype=torch.float)
    return {'w': w, 'b': b}


class GradientDescentWithAutoGradient(GradientDescent):
    def __init__(self, x, y, initial_parameters, loss: Loss, learning_rate: float):
        super(GradientDescentWithAutoGradient, self).__init__(
            x, y, initial_parameters, loss, learning_rate)

    def forward(self):
        return h(self.x, self.parameters)

    def backward(self, loss):
        # backward with autogradient
        loss.backward()

        # In PyTorch, every method that ends with an underscore (_) makes changes in-place,
        # meaning, they will modify the underlying variable.
        # We need to use NO_GRAD to keep the update out of the gradient computation
        # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
        with torch.no_grad():
            super(GradientDescentWithAutoGradient, self).backward(loss)

        # If you check the method’s documentation,
        # it clearly states that gradients are accumulated.
        # So, every time we use the gradients to update the parameters,
        # we need to zero the gradients afterwards.
        # And that’s what zero_() is good for.
        for param in self.parameters:
            self.parameters[param].grad.zero_()

    def grad(self):
        return {"w": self.parameters["w"].grad, "b": self.parameters["b"].grad}


def show_demo():
    utils.ensure_reproducity()
    # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    m = 1000
    x = torch.randn(m, dtype=torch.float)
    true_w = 197
    true_b = 769
    noise = 0.01 * torch.randn(m, dtype=torch.float)
    y = true_w * x + true_b + noise

    plt.figure(figsize=(10, 5))
    for learning_rate in np.logspace(-2, 0, 10):
        parameters = init_params()
        gd = GradientDescentWithAutoGradient(
            x, y, parameters, MSELoss(), learning_rate)
        plt.plot(gd.train(), label=str(learning_rate))
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend(loc='upper left')


if __name__ == "__main__":
    show_demo()
