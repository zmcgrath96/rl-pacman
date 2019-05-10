import numpy as np 
from utils import *

class nn:
	def __init__(self, num_input, num_output, hidden=100):
		self.w1 = np.random.rand(hidden, num_input) / np.sqrt(num_input)
		self.w2 = np.random.rand(hidden) / np.sqrt(hidden)

	# forward pass through the data
	# returns the probablity of doing action and hidden state
	def forward(self, x):
		h = np.dot(self.w1, x)
		h[h<0] = 0
		p = sigmoid(np.dot(self.w2, h))
		return p, h 

	def backward():
		pass 

	