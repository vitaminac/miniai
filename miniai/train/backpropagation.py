class BackpropagationWithAutoGrad():
    def __init__(self, x, y, model: callable, loss: callable):
        self.x = x
        self.y = y
        self.model = model
        self.loss = loss

    def __call__(self, parameters):
        loss = self.forward(parameters)
        return self.backward(loss, parameters)

    def forward(self, parameters):
        # we need to zero the previous accumulated gradients.
        self._zero_grad(parameters)

        y_hat = self.model(self.x, parameters)
        loss = self.loss(self.y, y_hat)
        return loss

    def backward(self, loss, parameters):
        # backward with autogradient
        loss.backward()
        parameters_grad = {}
        for param in parameters:
            parameters_grad[param] = parameters[param].grad
        return parameters_grad

    def _zero_grad(self, parameters):
        # If you check the method’s documentation,
        # it clearly states that gradients are accumulated.
        # To zero the accumulated gradients we use zero_()
        for param in parameters:
            gradient = parameters[param].grad
            if gradient is not None:
                gradient.zero_()
