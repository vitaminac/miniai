from abc import abstractmethod
from ..loss import Loss
from .trainable import Trainable


class GradientDescent(Trainable):
    def __init__(self, x, y, initial_parameters, loss: Loss, learning_rate: float) -> None:
        super(GradientDescent, self).__init__(
            x, y, initial_parameters, loss)
        self.learning_rate = learning_rate

    def backward(self, loss):
        grad = self.grad()
        for key in self.parameters:
            # perform the update using the gradient
            self.parameters[key] -= self.learning_rate * grad[key]

    @abstractmethod
    def grad(self):
        pass
