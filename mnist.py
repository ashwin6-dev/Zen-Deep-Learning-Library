from sklearn.datasets import load_digits
import numpy as np
import nn
import optim
import loss
from autodiff import *

def one_hot(n, max):
    arr = [0] * max

    arr[n - 1] = 1

    return arr


mnist = load_digits()
images = np.array([image.flatten() for image in mnist.images])
targets = np.array([one_hot(n, 10) for n in mnist.target])


model = nn.Model([
    nn.Linear(64),
    nn.Linear(32),
    nn.Sigmoid(),
    nn.Linear(10),
    nn.Softmax()
])

model.train(images[:1000], targets[:1000], epochs=150, loss_fn=loss.categorical_crossentropy, optimizer=optim.RMSProp(0.001), batch_size=128)

idx = 0
right = 0

for image in images:
    pred = (np.argmax(model(np.array([image])).value, axis=1) + 1) % 10
    print (pred, mnist.target[idx])
    if pred[0] == mnist.target[idx]:
        right += 1
    idx += 1

print ("accuracy:", right / idx)