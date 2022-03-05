import autodiff as ad
import nn
import numpy as np
import optim
import loss
import sys

sys.setrecursionlimit(10**6)
def fizz_buzz_encode(x):
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]
        
def dataset():
    x = []
    y = []

    for i in range(70):
        x.append([i >> j & 1 for j in range(10)])
        y.append(fizz_buzz_encode(i))

    return np.array(x), np.array(y)



n = nn.Net([
    nn.Linear(50),
    nn.Linear(25),
    nn.Sigmoid(),
    nn.Linear(4),
    nn.Sigmoid(),
])

x, y = dataset()

n.train(x, y, epochs=1000, optimizer=optim.RMSProp(lr=0.02), loss_fn=loss.MSE)

for i in range(0, 100):
    z = np.array([[i >> j & 1 for j in range(10)]])
    pred = n(z)
    idx = np.argmax(pred.value, axis=1)[0]
    real = np.argmax(fizz_buzz_encode(i))

    print ([i, "fizz","buzz","fizzbuzz"][idx], [i,"fizz","buzz","fizzbuzz"][real])