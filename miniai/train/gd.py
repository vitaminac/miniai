class GradientDescent():
    def __init__(self, f_grad, initial_parameters, learning_rate: float) -> None:
        super().__init__()
        self.f_grad = f_grad
        self.parameters = initial_parameters
        self.learning_rate = learning_rate

    def train(self, iterations: int = 20):
        for epoch in range(iterations):
            self.improve()
            yield epoch

    def improve(self):
        grad = self.f_grad(self.parameters)
        self.update(grad)

    def update(self, parameters_grad):
        for param in self.parameters:
            # perform the update using the gradient
            self.parameters[param] -= self.learning_rate * \
                parameters_grad[param]
