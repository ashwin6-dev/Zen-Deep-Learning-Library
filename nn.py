import autodiff as ad
import numpy as np
import loss 
import optim

np.random.seed(2)

class Layer:
    def __init__(self):
        pass

class Linear(Layer):
    def __init__(self, units):
        self.units = units
        self.w = None
        self.b = None

    def __call__(self, x):
        if self.w is None:
            self.w = ad.Tensor(np.random.randn(x.shape[-1], self.units))
            self.b = ad.Tensor(np.random.randn(1, self.units))

        return x @ self.w + self.b

class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.e ** (-1 * x))

class Softmax:
    def __call__(self, x):
        return (np.e ** (x - np.max(x.value))) / ad.reduce_sum(np.e ** (x - np.max(x.value)))

class Tanh:
    def __call__(self, x):
        pos = np.e ** x
        neg = np.e ** (-1*x)

        return (pos - neg) / (pos + neg)

class Net:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)

        return output

    def train(self, x, y, epochs=10, loss_fn = loss.MSE, optimizer=optim.SGD(lr=0.1)):
        for epoch in range(epochs):
            output = self(x)
            l = loss_fn(output, y)
            optimizer(self, l)
            
            print (epoch, l)