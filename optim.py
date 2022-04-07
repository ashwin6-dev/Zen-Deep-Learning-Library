import jax.numpy as np
from jax import jit 

def GradientDescent(lr=0.01):

    def fn(grads, params):
        new_params = []
        for grad_group, param_group in zip(grads, params):
            new_param_group = []
            for grad, param in zip(grad_group, param_group):
                new_param_group.append(param - (grad * lr))
            new_params.append(new_param_group)
        return new_params
    return jit(fn)

class RMSprop:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.averages = []

    def __call__(self, grads, params):
        self.averages = [[0] * len(param) for param in params] if len(self.averages) == 0 else self.averages
        new_params = []
        group = 0
        for grad_group, param_group in zip(grads, params):
            new_param_group = []
            param_idx = 0
            for grad, param in zip(grad_group, param_group):
                avg = (self.beta * self.averages[group][param_idx]) + (1 - self.beta) * grad**2
                new_param_group.append(param - ((self.lr / (np.sqrt(avg + 1e-8))) * grad))
                self.averages[group][param_idx] = avg
                param_idx += 1
            new_params.append(new_param_group)
            group += 1
        return new_params

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.averages1 = []
        self.averages2 = []

    def __call__(self, grads,params):
        self.averages1 = [[0] * len(param) for param in params] if len(self.averages1) == 0 else self.averages1
        self.averages2 = [[0] * len(param) for param in params] if len(self.averages2) == 0 else self.averages2
        new_params = []
        group = 0
        for grad_group, param_group in zip(grads, params):
            new_param_group = []
            param_idx = 0
            
            for grad, param in zip(grad_group, param_group):
                avg1 = (self.beta1 * self.averages1[group][param_idx]) + ((1 - self.beta1) * grad) / (1 - self.beta1)
                avg2 = (self.beta2 * self.averages2[group][param_idx]) + ((1 - self.beta2) * grad**2) / (1 -self.beta2)

                new_param_group.append(param - (self.lr * (avg1 / (np.sqrt(avg2) + 1e-8))))

                self.averages1[group][param_idx] = avg1
                self.averages2[group][param_idx] = avg2
                param_idx += 1
            new_params.append(new_param_group)
            group += 1
        return new_params