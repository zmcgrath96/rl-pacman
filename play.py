import sys
import numpy as np
from utils import process_frame
from nn import nn

def train():
	FRAME_DIMS = (300, 400)
	hidden = 300		# hidden layers in model
	batch_size = 10		# size of batch
	lr = .001			# learing rate
	gamma = .99			# discount factor for reward
	decay_rate = .99
	epsilon = .2		# starting value for exploration
	epochs = 100		# how many sets of batches to go through
	last_frame = np.zeros(FRAME_DIMS)
	num_actions =  4	# number of different actions that can be taken

	# setup the neural net
	net = nn(FRAME_DIMS[0] * FRAME_DIMS[1], num_actions, hidden)

	for epoch in range(epochs):
		print('Epoch: {}'.format(epoch))
		for batch in range(batch_size):
			game = True
			observations, states, loss_logs = [], [], []
			while game:
				# TODO: SOMETHING WITH FRAME
				# get the difference between the last frame and the current frame
				# set the last frame to the current frame
				frame = np.zeros(FRAME_DIMS)
				diff_frame = process_frame(frame) - process_frame(last_frame)
				last_frame = frame

				# action_prob should be of form [up, down, left, right] for pacman
				# decide if we should explore or if we should take the advice of our network
				action_prob, hidden_state = net.forward(diff_frame)
				explore = float(np.random.randint(1, 100) / 100) <= epsilon
				action = np.argmax(action_prob) if not explore else np.random.choice(len(action_prob))

				# keep track of observations and states for back propagation
				observations.append(diff_frame)
				states.append(hidden_state)

				# keep track of 'losses'. its a bit different for RL as the data isn't labeled
				loss_logs.append(action - action_prob[action])

				# get the reward from the environment

def play(): 
	pass

if __name__ == '__main__':
	if len(sys.argv) > 1 and '-t' in sys.argv[1]:
		train()
	else:
		play()