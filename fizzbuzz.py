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



n = nn.Model([
    nn.Linear(50),
    nn.Tanh(),
    nn.Linear(25),
    nn.Sigmoid(),
    nn.Linear(4),
    nn.Softmax(),
])

x, y = dataset()

n.train(x, y, epochs=150, optimizer=optim.RMSProp(lr=0.01), loss_fn=loss.categorical_crossentropy)


correct = 0
cnt = 0
for i in range(0, 100):
    z = np.array([[i >> j & 1 for j in range(10)]])
    pred = n(z)
    idx = np.argmax(pred.value, axis=1)[0]
    real = np.argmax(fizz_buzz_encode(i))

    pred = [i, "fizz","buzz","fizzbuzz"][idx]
    real = [i,"fizz","buzz","fizzbuzz"][real]

    if pred == real:
        correct += 1

    print (pred, real)
    cnt += 1

print ("accuracy: ", correct / cnt)