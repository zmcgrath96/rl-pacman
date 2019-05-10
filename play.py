import sys
import numpy as np

def train():
	hidden = 300		# hidden layers in model
	batch_size = 10		# size of batch
	lr = .001			# learing rate
	gamma = .99			# discount factor for reward
	decay_rate = .99
	pass

def play(): 
	pass

if __name__ == '__main__':
	if len(sys.argv) > 1 and '-t' in sys.argv[1]:
		train()
	else:
		play()