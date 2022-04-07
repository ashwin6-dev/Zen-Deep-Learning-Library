from sklearn.datasets import load_digits
import numpy as np
import nn
import optim
import loss

def one_hot(n, max):
    arr = [0] * max

    arr[n - 1] = 1

    return arr


mnist = load_digits()
images = np.array([image.flatten() for image in mnist.images])
targets = np.array([one_hot(n, 10) for n in mnist.target])

model = nn.Model([
    nn.Linear(100, input_shape=(None, 8*8), activation=nn.sigmoid),
    nn.Linear(50, input_shape=(None, 100), activation=nn.sigmoid),
    nn.Linear(10, input_shape=(None, 50), activation=nn.softmax)
])

model.train(images, targets, epochs=100, loss=loss.CategoricalCrossentropy, optim=optim.Adam(0.01), batch_size=10000)

idx = 0
for image in images:
    pred = (np.argmax(model.predict(np.array([image])), axis=1) + 1) % 10
    print (pred, mnist.target[idx])
    idx += 1

