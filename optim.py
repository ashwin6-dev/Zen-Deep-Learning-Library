class SGD:
    def __init__(self, lr):
        self.lr = lr

    def delta(self, param):
        return param.gradient * self.lr

    def __call__(self, model, loss):
        from nn import Layer
        loss.get_gradients()

        for layer in model.layers:
            if isinstance(layer, Layer):
                layer.update(self)

class Momentum:
    def __init__(self, lr = 0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.averages = {}

    def momentum_average(self, prev, grad):
        return (self.beta * prev) + (self.lr * grad)

    def delta(self, param):
        param_id = param.id

        if param_id not in self.averages:
            self.averages[param_id] = 0

        self.averages[param_id] = self.momentum_average(self.averages[param_id], param.gradient)
        return self.averages[param_id]

    def __call__(self, model, loss):
        from nn import Layer
        loss.get_gradients()
        for layer in model.layers:
            if isinstance(layer, Layer):
                layer.update(self)

class RMSProp:
    def __init__(self, lr = 0.01, beta=0.9, epsilon=10**-10):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.averages = {}

    def rms_average(self, prev, grad):
        return self.beta * prev + (1 - self.beta) * (grad ** 2)

    def delta(self, param):
        param_id = param.id

        if param_id not in self.averages:
            self.averages[param_id] = 0

        self.averages[param_id] = self.rms_average(self.averages[param_id], param.gradient)
        return (self.lr / (self.averages[param_id] + self.epsilon) ** 0.5) * param.gradient

    def __call__(self, model, loss):
        from nn import Layer
        loss.get_gradients()
        for layer in model.layers:
            if isinstance(layer, Layer):
                layer.update(self)
class Adam:
    def __init__(self, lr = 0.01, beta1=0.9, beta2=0.999, epsilon=10**-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.averages = {}
        self.averages2 = {}

    def rms_average(self, prev, grad):
        return (self.beta2 * prev) + (1 - self.beta2) * (grad ** 2)

    def momentum_average(self, prev, grad):
        return (self.beta1 * prev) + ((1 - self.beta1) * grad)

    def delta(self, param):
        param_id = param.id

        if param_id not in self.averages:
            self.averages[param_id] = 0
            self.averages2[param_id] = 0

        self.averages[param_id] = self.momentum_average(self.averages[param_id], param.gradient)
        self.averages2[param_id] = self.rms_average(self.averages2[param_id], param.gradient)

        adjust1 = self.averages[param_id] / (1 - self.beta1)
        adjust2 = self.averages2[param_id] / (1 - self.beta2)


        return self.lr * (adjust1 / (adjust2 ** 0.5 + self.epsilon))

    def __call__(self, model, loss):
        from nn import Layer
        loss.get_gradients()
        for layer in model.layers:            
            if isinstance(layer, Layer):
                layer.update(self)