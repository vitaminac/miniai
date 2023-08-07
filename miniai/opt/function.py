from abc import ABC, abstractmethod


class Function(ABC):
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError()


class DifferentiableFunction(Function):
    @abstractmethod
    def grad(self, x):
        raise NotImplementedError()
