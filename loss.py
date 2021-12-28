import numpy as np

class MSE:
    def __call__(self, error):
        self.error = error
        return np.mean(error ** 2)
    
    def backward(self):
        return 2 * self.error 