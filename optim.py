
class SGD:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, model, loss):
        from nn import Layer
        loss.get_gradients()
        for layer in model.layers:
            if isinstance(layer, Layer):
                w_d = layer.w.gradient
                b_d = layer.b.gradient

                layer.w = layer.w - (w_d * self.lr)
                layer.b = layer.b - (b_d * self.lr)

class Momentum:
    def __init__(self, lr = 0.01, beta=0.9):
        self.lr = lr
        self.beta = beta

    def momentum_average(self, prev, grad):
        return (self.beta * prev) + (self.lr * grad)

    def __call__(self, model, loss):
        from nn import Layer
        loss.get_gradients()
        for layer in model.layers:
            if isinstance(layer, Layer):
                w_d = layer.w.gradient
                b_d = layer.b.gradient
                if not hasattr(layer, "momentum"):
                    layer.momentum = {
                        "w": 0,
                        "b": 0
                    }

                layer.momentum["w"] = self.momentum_average(layer.momentum["w"], w_d)
                layer.momentum["b"] = self.momentum_average(layer.momentum["b"], b_d)

                layer.w -= layer.momentum["w"]
                layer.b -= layer.momentum["b"]

class RMSProp:
    def __init__(self, lr = 0.01, beta=0.9, epsilon=10**-10):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

    def rms_average(self, prev, grad):
        return self.beta * prev + (1 - self.beta) * (grad ** 2)

    def __call__(self, model, loss):
        from nn import Layer
        loss.get_gradients()

        for layer in model.layers:
            if isinstance(layer, Layer):
                w_d = layer.w.gradient
                b_d = layer.b.gradient
                if not hasattr(layer, "rms"):  
                    layer.rms = {
                        "w": 0,
                        "b": 0
                    }

                layer.rms["w"] = self.rms_average(layer.rms["w"], w_d)
                layer.rms["b"] = self.rms_average(layer.rms["b"], b_d)

                layer.w -= (self.lr / (layer.rms["w"] + self.epsilon) ** 0.5) * w_d
                layer.b -= (self.lr / (layer.rms["b"] + self.epsilon) ** 0.5) * b_d

class Adam:
    def __init__(self, lr = 0.01, beta1=0.9, beta2=0.999, epsilon=10**-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def rms_average(self, prev, grad):
        return (self.beta2 * prev) + (1 - self.beta2) * (grad ** 2)

    def momentum_average(self, prev, grad):
        return (self.beta1 * prev) + ((1 - self.beta1) * grad)

    def __call__(self, model, loss):
        from nn import Layer
        loss.get_gradients()
        for layer in model.layers:            
            if isinstance(layer, Layer):
                w_d = layer.w.gradient
                b_d = layer.b.gradient
                if not hasattr(layer, "adam"):
                    layer.adam = {
                        "w": 0,
                        "b": 0,
                        "w2": 0,
                        "b2": 0
                    }

                layer.adam["w"] = self.momentum_average(layer.adam["w"], w_d)
                layer.adam["b"] = self.momentum_average(layer.adam["b"], b_d)
                layer.adam["w2"] = self.rms_average(layer.adam["w2"], w_d)
                layer.adam["b2"] = self.rms_average(layer.adam["b2"], b_d)

                w_adjust = layer.adam["w"] / (1 - self.beta1)
                b_adjust = layer.adam["b"] / (1 - self.beta1)
                w2_adjust = layer.adam["w2"] / (1 - self.beta2)
                b2_adjust = layer.adam["b2"] / (1 - self.beta2)

                

                layer.w -= self.lr * (w_adjust / (w2_adjust ** 0.5 + self.epsilon))
                layer.b -= self.lr * (b_adjust / (b2_adjust ** 0.5 + self.epsilon))