from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from ..loss import Loss


class Trainable(ABC):
    def __init__(self, x, y, initial_parameters, loss: Loss) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.parameters = initial_parameters
        self.loss_function = loss

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, loss):
        pass

    def train(self, iterations: int = 20):
        writer = SummaryWriter()
        losses = []
        for i in range(iterations):
            # Forward pass: compute predicted y
            y_hat = self.forward()

            # Compute loss
            loss = self.loss_function(self.y, y_hat)

            writer.add_scalar("Loss/train", loss, i)
            losses.append(loss.item())

            self.backward(loss)
        writer.flush()
        writer.close()
        return losses
