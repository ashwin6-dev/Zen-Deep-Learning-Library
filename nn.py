import jax.numpy as np
import functools
from jax import grad, random, value_and_grad, vmap, jit
from loss import *
from optim import *

seed = random.PRNGKey(345)

class Layer:
    def __init__(self):
        pass

class Linear(Layer):
    def __init__(self, units, input_shape=None, activation = lambda x : x):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation

    def forward(self, inp, params):
        W = params[0]
        b = params[1]

        if self.input_shape is None:
            self.input_shape = inp.shape

        return self.activation(np.matmul(inp, W) + b)

    def init_params(self):
        return [random.uniform(seed, (self.input_shape[-1], self.units), minval=-1/np.sqrt(self.input_shape[-1]), maxval=1/np.sqrt(self.input_shape[-1])), np.zeros((1, self.units))]

class RNN(Layer):
    def __init__(self, units, input_shape=None, hidden_dim=64, activation = lambda x : x):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.Wh = Linear(hidden_dim, input_shape=(None, input_shape[-1] + hidden_dim), activation=tanh)
        self.Wy = Linear(units, input_shape=(None, hidden_dim), activation=activation)
        
    def one_forward(self, x, params):
        state = np.zeros((1, self.hidden_dim))
        
        for time_step in x:
            time_step = np.expand_dims(time_step, axis=0)
            concat = np.concatenate([time_step, state], axis=1)
            state = self.Wh.forward(concat, params)

        return state
        
    def forward(self, x, params):
        vec_one_forward = vmap(self.one_forward, in_axes=[0, None], out_axes=0)
        final_state = np.sum(vec_one_forward(x, params[0]), axis=1)

        return self.Wy.forward(final_state, params[1])

    def init_params(self):
        return self.Wh.init_params(), self.Wy.init_params()

class Model:
    def __init__(self, layers):
        self.layers = layers
        self.initialised = False
        self.params = []

    def initialise(self):
        if not self.initialised:
            for layer in self.layers:
                params = layer.init_params()
                if isinstance(layer, RNN):
                    self.params.append(params[0])
                    self.params.append(params[1])
                else:
                    self.params.append(params)

            self.initialised = True

    
    def get_loss(self, x, y, loss_fn, params):
        pred = self.predict(x, params)
        loss = loss_fn(y, pred)

        return loss

    #@functools.partial(jit, static_argnums=(0, 3, 4, 5, 6))
    def train(self, x, y, epochs=10, loss=MeanSquaredError, optim=GradientDescent(lr=0.1), batch_size=32):
        for epoch in range(epochs):
            epoch_loss = 0
            batches = 0
            for batch in range(0, len(x), batch_size):
                _loss = self.get_loss(x[batch:batch+batch_size], y[batch:batch+batch_size], loss, self.params)
                grads = grad(self.get_loss, -1)(x[batch:batch+batch_size], y[batch:batch+batch_size], loss, self.params)
                self.params = optim(grads, self.params)
                epoch_loss += _loss
                batches += 1
            print (epoch+1, epoch_loss / batches)
            
    def predict(self, x, params=None):
        self.initialise()

        params = self.params if params is None else params

        pred = x
        idx = 0
        param_idx = 0
        while idx < len(self.layers):
            layer, _params = self.layers[idx], params[param_idx]
            if isinstance(layer, RNN):
                pred = layer.forward(pred, [_params, params[param_idx+1]])
                param_idx += 2
            else:
                pred = layer.forward(pred, _params)
                param_idx += 1
            idx += 1
            

        return pred

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, np.zeros_like(x))

def softmax(x):
    max = np.max(x,axis=1,keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x,axis=1,keepdims=True)
    f_x = e_x / sum 
    return f_x