import numpy as np
import math
import nn
import optim
import loss
from matplotlib import pyplot as plt

x_axis = np.array(list(range(200))).T
seq = [math.sin(i) * math.cos(i) for i in range(200)]

x = []
y = []

for i in range(len(seq) - 50):
    new_seq = [[i] for i in seq[i:i+50]]
    x.append(new_seq)
    y.append([seq[i+50]])

model = nn.Model([
    nn.RNN(15, input_shape=(50, 1), hidden_dim=64),
    nn.Linear(10, input_shape=(None, 15)),
    nn.Linear(1, input_shape=(None, 10))
])

model.train(np.array(x[:50]), np.array(y[:50]), epochs=300, optim=optim.Adam(0.005), loss=loss.MeanSquaredError, batch_size=75)

preds = []

for i in x:
    preds.append(model.predict(np.expand_dims(np.array(i), axis=0))[0][0])

plt.plot(x_axis[:150], seq[:150])
plt.plot(x_axis[:150], preds)
plt.savefig("matplotlib.png")