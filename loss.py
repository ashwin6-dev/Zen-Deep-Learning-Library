import jax.numpy as np

def MeanSquaredError(real, pred):
    return np.mean((real - pred)**2)

def RootMeanSquaredError(real, pred):
    return np.sqrt(np.mean((real - pred)**2))

def CategoricalCrossentropy(real, pred):
    return -np.mean(real * np.log(pred))