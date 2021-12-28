import layers
import numpy as np


x = np.array([[0, 1], [0, 0], [1, 1], [0, 1]])
y = np.array([[1], [0], [0], [1]])

net = layers.Model([
    layers.Linear(8),
    layers.Linear(4),
    layers.Sigmoid(),
    layers.Linear(1),
    layers.Sigmoid()
])

print (net(x))