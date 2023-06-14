from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def __call__(self, y, y_hat):
        pass

    @abstractmethod
    def derivative(self, y, y_hat):
        pass
