#optim.py
import numpy as np
import layers
import tqdm

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def __call__(self, model, loss):
        grad = loss.backward()

        for layer in tqdm.tqdm(model.layers[::-1]):
            grad = layer.backward(grad)
            
            if isinstance(layer, layers.Layer):
                layer.w -= layer.w_gradient * self.lr
                layer.b -= layer.b_gradient * self.lr

class Momentum:
    def __init__(self, lr = 0.01, beta=0.9):
        self.lr = lr
        self.beta = beta

    def momentum_average(self, prev, grad):
        return (self.beta * prev) + (self.lr * grad)

    def __call__(self, model, loss):
        grad = loss.backward()

        for layer in tqdm.tqdm(model.layers[::-1]):
            grad = layer.backward(grad)
            
            if isinstance(layer, layers.Layer):
                if not hasattr(layer, "momentum"):
                    layer.momentum = {
                        "w": 0,
                        "b": 0
                    }

                layer.momentum["w"] = self.momentum_average(layer.momentum["w"], layer.w_gradient)
                layer.momentum["b"] = self.momentum_average(layer.momentum["b"], layer.b_gradient)

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
        grad = loss.backward()

        for layer in tqdm.tqdm(model.layers[::-1]):
            grad = layer.backward(grad)
            
            if isinstance(layer, layers.Layer):
                if not hasattr(layer, "rms"):
                    layer.rms = {
                        "w": 0,
                        "b": 0
                    }

                layer.rms["w"] = self.rms_average(layer.rms["w"], layer.w_gradient)
                layer.rms["b"] = self.rms_average(layer.rms["b"], layer.b_gradient)

                layer.w -= self.lr / (np.sqrt(layer.rms["w"] + self.epsilon)) * layer.w_gradient
                layer.b -= self.lr / (np.sqrt(layer.rms["b"] + self.epsilon)) * layer.b_gradient
                
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
        grad = loss.backward()

        for layer in tqdm.tqdm(model.layers[::-1]):
            grad = layer.backward(grad)
            
            if isinstance(layer, layers.Layer):
                if not hasattr(layer, "adam"):
                    layer.adam = {
                        "w": 0,
                        "b": 0,
                        "w2": 0,
                        "b2": 0
                    }

                layer.adam["w"] = self.momentum_average(layer.adam["w"], layer.w_gradient)
                layer.adam["b"] = self.momentum_average(layer.adam["b"], layer.b_gradient)
                layer.adam["w2"] = self.rms_average(layer.adam["w2"], layer.w_gradient)
                layer.adam["b2"] = self.rms_average(layer.adam["b2"], layer.b_gradient)

                w_adjust = layer.adam["w"] / (1 - self.beta1)
                b_adjust = layer.adam["b"] / (1 - self.beta1)
                w2_adjust = layer.adam["w2"] / (1 - self.beta2)
                b2_adjust = layer.adam["b2"] / (1 - self.beta2)

                layer.w -= self.lr * (w_adjust / np.sqrt(w2_adjust) + self.epsilon)
                layer.b -= self.lr * (b_adjust / np.sqrt(b2_adjust) + self.epsilon)

