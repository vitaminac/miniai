from . import loss


class MSELoss(loss.Loss):
    def __init__(self):
        pass

    def __call__(self, y, y_hat):
        return ((y_hat - y)**2).mean()

    def derivative(self, y, y_hat):
        return y_hat - y
