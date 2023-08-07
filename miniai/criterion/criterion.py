from abc import ABC, abstractmethod


class Criterion(ABC):
    @abstractmethod
    def __call__(self, y, y_hat):
        raise NotImplementedError()

    @abstractmethod
    def grad(self, y, y_hat):
        raise NotImplementedError()
