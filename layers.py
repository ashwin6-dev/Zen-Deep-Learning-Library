import numpy as np

class Model: 
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)

        return output

    def train(self, x, y, optim, loss, epochs=10):
        for epoch in range(1, epochs + 1):
            pred = self.__call__(x)
            error = y - pred
            l = loss(error)
            optim(self, l)
            print (f"epoch {epoch} loss {l}")

class Linear:
    def __init__(self, units):
        self.units = units
        self.w = False
        self.b = False

    def __call__(self, x):
        self.input = x
        if not self.w:
            self.w = np.random.randn(self.input.shape[-1], self.units)
        if not self.b:
            self.b = np.random.randn(self.units)

        return self.input @ self.w + self.b

    def backward(self, grad):
        self.w_gradient = self.input.T @ grad
        self.b_gradient = grad
        return grad @ self.w.T

class Sigmoid:
    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-x))

        return self.output

    def backward(self, grad):
        return grad * (self.output * (1 - self.output))

class Relu:
    def __call__(self, x):
        self.output = np.maximum(0, x)   
        return self.output

    def backward(self, grad):
        return grad * np.clip(self.output, 0, 1)

class Softmax:
    def __call__(self, x):
        self.output = np.exp(x) / np.sum(np.exp(x))
        return self.output
    
    def backward(self, grad):
        reshape_output = self.output.reshape(-1, 1)

        return grad * (np.diagflat(reshape_output) - np.dot(reshape_output, reshape_output.T))

class Tanh:
    def __call__(self, x):
        self.output = np.tanh(x)

        return self.output

    def backward(self, grad):
        return grad * (1 - self.output ** 2)