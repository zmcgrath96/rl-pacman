import numpy as np

def sigmoid(x):
        return (1 / (1 + np.exp(-x)))

def sigmoid_deriv(x):
    return x * (1.0 - x)

# take in a frame of n, n, 3 for rgb images
# return a frame thats n/2, n/2, avg(rgb)
def process_frame(x):
	avg_color = np.sum(x, axis=2) / 3
	w, h = avg_color.shape
	k = 2
	wk = w / 2
	hk = h / 2
	return avg_color[:wk * k, :hk * k].reshape(wk, k, hk, k).max(axis=(1,3))

