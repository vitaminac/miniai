import torch
from .gd import GradientDescent
from .backpropagation import BackpropagationWithAutoGrad


class GradientDescentWithAutoGrad(GradientDescent):
    def __init__(self, backpropagation: BackpropagationWithAutoGrad, initial_parameters, learning_rate: float) -> None:
        super().__init__(backpropagation, initial_parameters, learning_rate)

    def update(self, parameters_grad):
        # In PyTorch, every method that ends with an underscore (_) makes changes in-place,
        # meaning, they will modify the underlying variable.
        # We need to use NO_GRAD to keep the update out of the gradient computation
        # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
        with torch.no_grad():
            super().update(parameters_grad)
