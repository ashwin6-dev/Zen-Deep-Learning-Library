import autodiff as ad
import numpy as np
import loss 
import optim
from tqdm import tqdm

np.random.seed(345)

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
            self.w = ad.Tensor(np.random.uniform(size=(x.shape[-1], self.units), low=-1/np.sqrt(x.shape[-1]), high=1/np.sqrt(x.shape[-1])))
            self.b = ad.Tensor(np.zeros((1, self.units)))

        return x @ self.w + self.b

    def update(self, optim):
        self.w.value -= optim.delta(self.w)
        self.b.value -= optim.delta(self.b)

        self.w.grads = []
        self.w.dependencies = []
        self.b.grads = []
        self.b.dependencies = []

class RNN(Layer):
    def __init__(self, units, hidden_dim, return_sequences=False):
        self.units = units
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.U = None
        self.W = None
        self.V = None

    def one_forward(self, x):
        x = np.expand_dims(x, axis=1)
        state = np.zeros((x.shape[-1], self.hidden_dim))
        y = []

        for time_step in x:
            mul_u = self.U(time_step[0])
            mul_w = self.W(state)
            state = Tanh()(mul_u + mul_w)

            if self.return_sequences:
                y.append(self.V(state))

        if not self.return_sequences:
            state.value = state.value.squeeze()
            return state

        return y

    def __call__(self, x):
        if self.U is None:
            self.U = Linear(self.hidden_dim)
            self.W = Linear(self.hidden_dim)
            self.V = Linear(self.units)

        if not self.return_sequences:
            states = []
            for seq in x:
                state = self.one_forward(seq)

                states.append(state)
            
            s = ad.stack(states)

            return s

        sequences = []
        for seq in x:
            out_seq = self.one_forward(seq)
            sequences.append(out_seq)

        return sequences

    def update(self, optim):
        self.U.update(optim) 
        self.W.update(optim)
        
        if self.return_sequences:
            self.V.update(optim)

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.filter_list = []

    def __call__(self, x):
        if len(self.filter_list) == 0:
            for _ in range(self.filters):
                filter_value = np.random.normal(size=self.kernel_size)
                self.filter_list.append(ad.Tensor(filter_value))
            
            outputs = []
            for _filter in self.filter_list:
                convolved = ad.convolve(_filter, self.strides, x)
                outputs.append(convolved)
            
            out = ad.stack(outputs)
            return out

class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.e ** (-1 * x))

class Softmax:
    def __call__(self, x):
        e_x = np.e ** (x - np.max(x.value))
        s_x = (e_x) / ad.reduce_sum(e_x, axis=1, keepdims=True)
        return s_x

class Tanh:
    def __call__(self, x):
        tanh_x = np.tanh(x.value)
        var = ad.Tensor(tanh_x)
        var.grads.append(1 - tanh_x ** 2)
        var.dependencies.append(x)

        return var
        

class Model:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)

        return output

    def train(self, x, y, epochs=10, loss_fn = loss.MSE, optimizer=optim.SGD(lr=0.1), batch_size=32):
        for epoch in range(epochs):
            _loss = 0
            print ("**")
            print (" ")
            print (f"EPOCH", epoch + 1)
            for batch in tqdm(range(0, len(x), batch_size)):
                output = self(x[batch:batch+batch_size])
                l = loss_fn(output, y[batch:batch+batch_size])
                optimizer(self, l)
                _loss += l
            
            print ("LOSS", _loss.value)
            print (" ")