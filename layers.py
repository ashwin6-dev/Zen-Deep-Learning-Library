import numpy as np
import loss
import optim
np.random.seed(0)

class Activation:
    def __init__(self):
        pass

class Layer:
    def __init__(self):
        pass

class Model: 
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)

        return output

    def train(self, x, y, optim = optim.SGD(), loss=loss.MSE(), epochs=10):
        for epoch in range(1, epochs + 1):
            pred = self.__call__(x)
            error = pred - y
            l = loss(error)
            optim(self, loss)
            print (f"epoch {epoch} loss {l}")

class Linear(Layer):
    def __init__(self, units):
        self.units = units
        self.initialized = False

    def __call__(self, x):
        self.input = x
        if not self.initialized:
            self.w = np.random.randn(self.input.shape[-1], self.units)
            self.b = np.random.randn(self.units)
            self.initialized = True

        return self.input @ self.w + self.b

    def backward(self, grad):
        self.w_gradient = self.input.T @ grad
        self.b_gradient = np.sum(grad, axis=0)
        return grad @ self.w.T

class Sigmoid(Activation):
    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-x))

        return self.output

    def backward(self, grad):
        return grad * (self.output * (1 - self.output))

class Relu(Activation):
    def __call__(self, x):
        self.output = np.maximum(0, x)   
        return self.output

    def backward(self, grad):
        return grad * np.clip(self.output, 0, 1)

class Softmax(Activation):
    def __call__(self, x):
        self.output = np.exp(x) / np.sum(np.exp(x))
        return self.output
    
    def backward(self, grad):
        reshape_output = self.output.reshape(-1, 1)

        return grad * (np.diagflat(reshape_output) - np.dot(reshape_output, reshape_output.T))

class Tanh(Activation):
    def __call__(self, x):
        self.output = np.tanh(x)

        return self.output

    def backward(self, grad):
        return grad * (1 - self.output ** 2)