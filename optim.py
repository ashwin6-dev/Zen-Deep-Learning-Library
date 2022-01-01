#optim.py

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