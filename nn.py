import numpy as np 
from utils import *

class nn:
	def __init__(self, num_input, num_output, hidden=100, lr, decay_rate):
		self.w1 = np.random.rand(hidden, num_input) / np.sqrt(num_input)
		self.w2 = np.random.rand(hidden) / np.sqrt(hidden)
		self.dw1 = None
		self.dw2 = None
		self.lr =  lr
		self.decay_rate = decay_rate
		self.rms_error_dw1 = None
		self.rms_error_dw2 = None

	# forward pass through the data
	# returns the probablity of doing action and hidden state
	def forward(self, x):
		h = np.dot(self.w1, x)
		h[h<0] = 0
		p = sigmoid(np.dot(self.w2, h))
		return p, h 

	# reverse pass through the states and loss
	def backward(self, state, loss):
		self.dw2 = np.dot(state.T, loss)
		dh = np.dot(loss, self.w2)
		dh[<0] = 0
		self.dw1 = np.dot(dh, state)
		 
	def update(self):
		if self.rms_error_dw1 is None:
			self.rms_error_dw1 = np.zeros(self.dw1.shape)
		if self.rms_error_dw2 is None:
			self.rms_error_dw2 = np.zeros(self.dw2.shape)

		self.rms_error_dw1 = self.decay_rate * self.rms_error_dw1 + (1 - self.decay_rate) * self.dw1**2
		self.rms_error_dw2 = self.decay_rate * self.rms_error_dw2 + (1 - self.decay_rate) * self.dw2**2

		self.w1 += (lr * self.dw1) / (np.sqrt(self.rms_error_dw1))
		self.w2 += (lr * self.dw2) / (np.sqrt(self.rms_error_dw2))

	