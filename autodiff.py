import numpy as np
import string
import random

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


np.seterr(invalid='ignore')

def is_matrix(o):
    return type(o) == np.ndarray

def same_shape(s1, s2):
    for a, b in zip(s1, s2):
        if a != b:
            return False

    return True

class Tensor:
    __array_priority__ = 1000
    def __init__(self, value, trainable=True):
        self.value = value
        self.dependencies = []
        self.grads = []
        self.grad_value = None
        self.shape = 0
        self.matmul_product = False
        self.gradient = 0
        self.trainable = trainable
        self.id = id_generator()

        if is_matrix(value):
            self.shape = value.shape

    def depends_on(self, target):
        if self == target:
            return True

        dependencies = self.dependencies

        for dependency in dependencies:
            if dependency == target:
                return True
            elif dependency.depends_on(target):
                return True

        return False

    def __mul__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)

        var = Tensor(self.value * other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(other.value)
        var.grads.append(self.value)
        return var

    def __rmul__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)
            
        var = Tensor(self.value * other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(other.value)
        var.grads.append(self.value)
        return var

    def __add__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)
            
        var = Tensor(self.value + other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(np.ones_like(self.value))
        var.grads.append(np.ones_like(other.value))
        return var

    def __radd__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)
            
        var = Tensor(self.value + other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(np.ones_like(self.value))
        var.grads.append(np.ones_like(other.value))
        return var

    def __sub__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other)
            
        var = Tensor(self.value - other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(np.ones_like(self.value))
        var.grads.append(-np.ones_like(other.value))
        return var

    def __rsub__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)
            
        var = Tensor(other.value - self.value)
        var.dependencies.append(other)
        var.dependencies.append(self)
        var.grads.append(np.ones_like(other.value))
        var.grads.append(-np.one_like(self.value))
        return var

    def __pow__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)
            
        var = Tensor(self.value ** other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)

        grad_wrt_self = other.value * self.value ** (other.value - 1)
        var.grads.append(grad_wrt_self)

        grad_wrt_other = (self.value ** other.value) * np.log(self.value)
        var.grads.append(grad_wrt_other)

        return var

    def __rpow__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)
            
        var = Tensor(other.value ** self.value)
        var.dependencies.append(other)
        var.dependencies.append(self)

        grad_wrt_other = self.value * other.value ** (self.value - 1)
        var.grads.append(grad_wrt_other)
        
        grad_wrt_self = (other.value ** self.value) * np.log(other.value)
        var.grads.append(grad_wrt_self)
        
        return var


    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)
        
    def __matmul__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)
        
        if other.value.ndim == 1:
            other.value = np.expand_dims(other.value, axis=0)

        if self.value.ndim == 1:
            self.value = np.expand_dims(self.value, axis=0)
        
        var = Tensor(self.value @ other.value)
        var.dependencies.append(self)
        var.dependencies.append(other)
        var.grads.append(other.value.T)
        var.grads.append(self.value.T)

        var.matmul_product = True
        return var

    def __rmatmul__(self, other):
        if not (isinstance(other, Tensor)):
            other = Tensor(other, trainable=False)
        
        if other.value.ndim == 1:
            other.value = np.expand_dims(other.value, axis=0)

        if self.value.ndim == 1:
            self.value = np.expand_dims(self.value, axis=0)

        var = Tensor(other.value @ self.value)
        var.dependencies.append(other)
        var.dependencies.append(self)
        var.grads.append(self.value.T)
        var.grads.append(other.value.T)

        var.matmul_product = True

        return var

    def grad(self, target, grad = None):
        grad = self.value / self.value if grad is None else grad
        grad = np.float32(grad)

        if not self.depends_on(target):
            return 0

        if self == target:
            return grad

        final_grad = 0

        for dependency, _grad in zip(self.dependencies, self.grads):
            local_grad = np.float32(_grad) if dependency.depends_on(target) else 0
            
            if local_grad is not 0:
                if self.matmul_product:                
                    if dependency == self.dependencies[0]:
                        local_grad = grad @ local_grad
                    else:
                        local_grad = local_grad @ grad
                else:
                    if dependency.shape != 0 and not same_shape(grad.shape, local_grad.shape):
                        ndims_added = grad.ndim - local_grad.ndim
                        for _ in range(ndims_added):
                            grad = grad.sum(axis=0)
                        
                        for i, dim in enumerate(local_grad.shape):
                            if dim == 1:
                                grad = grad.sum(axis=i, keepdims=True)

                    local_grad *= grad
            
            final_grad += dependency.grad(target, local_grad)

        return final_grad

        
    def get_gradients(self, grad = None):
        grad = 1 if grad is None else grad
        grad = np.float32(grad)

        for dependency, _grad in zip(self.dependencies, self.grads):
            if dependency.trainable:
                local_grad = np.float32(_grad)
                
                if self.matmul_product:                
                    if dependency == self.dependencies[0]:
                        local_grad = grad @ local_grad
                    else:
                        local_grad = local_grad @ grad
                else:
                    if dependency.shape != 0 and not same_shape(grad.shape, local_grad.shape):
                        ndims_added = grad.ndim - local_grad.ndim
                        for _ in range(ndims_added):
                            grad = grad.sum(axis=0)
                            
                        for i, dim in enumerate(dependency.shape):
                            if dim == 1:
                                grad = grad.sum(axis=i, keepdims=True)
                    
                    local_grad = local_grad * np.nan_to_num(grad)
                    
                    if hasattr(dependency, "reshape_grad"):
                        local_grad = local_grad.reshape(dependency.reshape_grad)

                dependency.gradient += local_grad
                dependency.get_gradients(local_grad)

    def __repr__(self):
        return f"Tensor ({self.value})"

def stack(tensor_list):
    tensor_values = [tensor.value for tensor in tensor_list]
    s = np.stack(tensor_values)

    var = Tensor(s)
    var.dependencies += tensor_list
    
    for tensor in tensor_list:
        var.grads.append(np.ones(tensor.value.shape))

    return var
    

def reduce_sum(tensor, axis = None, keepdims=False):
    var = Tensor(np.sum(tensor.value, axis = axis, keepdims=keepdims))
    var.dependencies.append(tensor)
    var.grads.append(np.ones(tensor.value.shape))

    return var

def reduce_mean(tensor, axis = None, keepdims=False):
    return reduce_sum(tensor, axis, keepdims) / tensor.value.size

def log(tensor):
    var = Tensor(np.log(tensor.value))
    var.dependencies.append(tensor)
    var.grads.append(1 / tensor.value)

    return var

def flatten(tensor):
    var = Tensor(tensor.value.flatten())
    var.dependencies.append(tensor)
    var.grads.append(np.ones_like(var.value))
    tensor.reshape_grad = tensor.value.shape

    return var

def reshape(tensor, shape):
    var = Tensor(tensor.value.reshape(*shape))
    var.dependencies.append(tensor)
    var.grads.append(np.ones_like(var.value))
    tensor.reshape_grad = tensor.value.shape

    return var
    
def conv2d(_filter, stride, x, padding = 0):
    if not (isinstance(x, Tensor)):
        x = Tensor(x, trainable=False)

    inp_shape = x.value.shape
    filter_shape = _filter.value.shape

    output_shape = ((np.asarray(inp_shape) +  2 * padding - (np.asarray(filter_shape) - 1) - 1) // stride) + 1
    output_shape = output_shape.tolist()

    windows = []
    window_indexes = []

    for row in range(0, inp_shape[0] - 1, stride):
        for col in range(0, inp_shape[1] - 1, stride):
            window = x.value[row : row + filter_shape[0], col : col + filter_shape[1]]
            windows.append(window.flatten())

            row_indexes = [i for i in range(row, row + filter_shape[0])]
            col_indexes = [i for i in range(col, col + filter_shape[1])]
            indexes = [ [r, c] for r in row_indexes for c in col_indexes ]
            window_indexes.append(indexes)

    windows = Tensor(np.array(windows).T)
    window_indexes = np.transpose(np.array(window_indexes), axes=(1, 0, 2)).tolist()
    flat_filter = flatten(_filter)
    
    x_grad = np.zeros_like(x.value)
    row_idx = 0
    for row in window_indexes:
        for col in row:
            x_row = col[0]
            x_col = col[1]
            x_grad[x_row][x_col] += flat_filter.value[row_idx]

        row_idx += 1
            

    feat_map = flat_filter @ windows
    feat_map = reshape(feat_map, output_shape)
    feat_map.dependencies = [x, _filter]
    feat_map.grads = [x_grad, windows.value.T.sum(axis=1, keepdims=True).reshape(filter_shape)]

    return feat_map