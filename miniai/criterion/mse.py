from .criterion import Criterion


class MSE(Criterion):
    def __call__(self, y, y_hat):
        return ((y_hat - y) ** 2).mean()

    def grad(self, y, y_hat):
        return y_hat - y
