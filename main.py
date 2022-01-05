import layers
import loss
import optim
import numpy as np


x = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])
y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]]) 

net = layers.Model([
    layers.Linear(8),
    layers.Linear(4),
    layers.Sigmoid(),
    layers.Linear(2),
    layers.Softmax()
])

net.train(x, y, optim=optim.RMSProp(lr=0.02), loss=loss.MSE(), epochs=200)

layers.save(net, "model")


print (net(x))