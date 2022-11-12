from autodiff import *
import numpy as np

a = Tensor(np.array([[3, 9., 0], [2, 8, 1], [1, 4, 8]]))
f = Tensor(np.array([[8., 9], [4, 4]]))

c = conv2d(f, 1, a)

print (c)

c.get_gradients()
print (a.gradient)
