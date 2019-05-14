import numpy as np 
from utils import *

class nn:
	def __init__(self, num_input, num_output, lr, decay_rate, hidden=100):
		self.w1 = np.random.randn(hidden, num_input)
		self.w2 = np.random.randn(num_output, hidden)
		self.h = None
		self.dw1 = None
		self.dw2 = None
		self.lr =  lr
		self.decay_rate = decay_rate
		self.rms_error_dw1 = None
		self.rms_error_dw2 = None

	# forward pass through the data
	# returns the probablity of doing action and hidden state
	def forward(self, x):
		self.h = sigmoid(np.dot(self.w1, np.reshape(x, (x.shape[0] * x.shape[1], 1))))
		p = sigmoid(np.dot(self.w2, self.h))
		return p.ravel(), self.h 

	# reverse pass through the states and loss
	def backward(self, state, loss):
		d_out = sigmoid_deriv(np.array(loss))
		d_out = np.reshape(d_out, (d_out.shape[0], 1))
		self.dw2 = np.dot(d_out, self.h.T)
		dh = np.dot(self.w2.T, d_out)
		reshaped_state = np.reshape(state, (state.shape[0] * state.shape[1], 1))
		self.dw1 = np.dot(dh * sigmoid_deriv(self.h), reshaped_state.T)
		# print('dw1.shape: {}, dw2.shape: {}'.format(self.dw1.shape, self.dw2.shape))
		self.update()

	def update(self):
		if self.rms_error_dw1 is None:
			self.rms_error_dw1 = np.full(self.dw1.shape, 1)
		if self.rms_error_dw2 is None:
			self.rms_error_dw2 = np.full(self.dw2.shape, 1)

		self.rms_error_dw1 = self.decay_rate * self.rms_error_dw1 + (1 - self.decay_rate) * self.dw1**2
		self.rms_error_dw2 = self.decay_rate * self.rms_error_dw2 + (1 - self.decay_rate) * self.dw2**2

		self.w1 += (self.lr * self.dw1) / (np.sqrt(self.rms_error_dw1))
		self.w2 += (self.lr * self.dw2) / (np.sqrt(self.rms_error_dw2))


	