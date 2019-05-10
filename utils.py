import numpy as np

def sigmoid(x):
        return (1 / (1 + np.exp(-x)))

def sigmoid_deriv(x):
    return x * (1.0 - x)
