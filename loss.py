#loss.py

import numpy as np

class MSE:
    def __call__(self, pred, y):
        self.error = pred - y
        return np.mean(self.error ** 2)
    
    def backward(self):
        return 2 * (1 / self.error.shape[-1]) * self.error 

class CrossEntropy:
    def __call__(self, pred, y):
        pass