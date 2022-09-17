import numpy as np
import autodiff as ad

def MSE(pred, real):
    loss = ad.reduce_mean((pred - real)**2)
    return loss

def categorical_crossentropy(pred, real):
    loss = -1 * ad.reduce_mean(real * ad.log(pred))
    
    return loss