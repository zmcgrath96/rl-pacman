import numpy as np 
from utils import *

class nn:
	def __init__(self, num_input, num_output, lr, decay_rate, hidden=100):
		self.w1 = np.full((hidden, num_input), 1 / (hidden * num_input))
		self.w2 = np.full((num_output, hidden), 1 / (hidden * num_output))
		self.dw1 = None
		self.dw2 = None
		self.lr =  lr
		self.decay_rate = decay_rate
		self.rms_error_dw1 = None
		self.rms_error_dw2 = None

	# forward pass through the data
	# returns the probablity of doing action and hidden state
	def forward(self, x):
		h = np.dot(self.w1, np.reshape(x, (x.shape[0] * x.shape[1], 1)))
		h[h<0] = 0
		p = sigmoid(np.dot(self.w2, h))
		return p.ravel(), h 

	# reverse pass through the states and loss
	def backward(self, state, loss):
		d_out = sigmoid_deriv(loss)
		self.dw2 = np.dot(d_out, self.w2)
		dh = np.dot(self.w2.T, d_out)
		reshaped_state = np.reshape(state, (state.shape[0] * state.shape[1], 1))
		dh[dh<0] = 0
		self.dw1 = np.dot(dh,reshaped_state)

	def update(self):
		if self.rms_error_dw1 is None:
			self.rms_error_dw1 = np.zeros(self.dw1.shape)
		if self.rms_error_dw2 is None:
			self.rms_error_dw2 = np.zeros(self.dw2.shape)

		self.rms_error_dw1 = self.decay_rate * self.rms_error_dw1 + (1 - self.decay_rate) * self.dw1**2
		self.rms_error_dw2 = self.decay_rate * self.rms_error_dw2 + (1 - self.decay_rate) * self.dw2**2

		self.w1 += (self.lr * self.dw1) / (np.sqrt(self.rms_error_dw1))
		self.w2 += (self.lr * self.dw2) / (np.sqrt(self.rms_error_dw2))

	