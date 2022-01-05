#loss.py

import numpy as np

class MSE:
    def __call__(self, pred, y):
        self.error = pred - y
        return np.mean(self.error ** 2)
    
    def backward(self):
        return 2 * (1 / self.error.shape[-1]) * self.error 

class CategoricalCrossentropy:
    def __call__(self, pred, y):
        self.error = pred - y
        return -np.sum(np.log(pred) * y)

    def backward(self):
        return self.error