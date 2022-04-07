import nn
import optim
import loss
import numpy as np

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

    for i in range(100):
        x.append([i >> j & 1 for j in range(10)])
        y.append(fizz_buzz_encode(i))

    return np.array(x), np.array(y)

model = nn.Model([
    nn.Linear(25, (None, 10), activation=nn.sigmoid),
    nn.Linear(4, (None, 25), activation=nn.softmax),
])

x, y = dataset()

model.train(x, y, epochs=250, batch_size=64, loss=loss.CategoricalCrossentropy, optim=optim.Adam(0.05, 0.9))


for i in range(0, 100):
    z = np.array([[i >> j & 1 for j in range(10)]])
    pred = model.predict(z)
    idx = np.argmax(pred, axis=1)[0]
    real = np.argmax(fizz_buzz_encode(i))

    print ([i, "fizz","buzz","fizzbuzz"][idx], [i,"fizz","buzz","fizzbuzz"][real])