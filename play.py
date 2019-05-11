import sys
import numpy as np
from utils import process_frame
from nn import nn
from game import Game

# ===============================================================================================
#							Helper functions 
# ===============================================================================================
def reward_reduction(rewards, gamma):
	reduced = [0 for _ in range(len(rewards))]
	running_add = 0
	# increase the rewards for things that happened towards the end of the game
	for t in range(len(rewards)):
		running_add =  running_add * gamma + rewards[t]
		reduced[t] = running_add

	return np.asarray(reduced)

# ===============================================================================================
#							END Helper functions 
# ===============================================================================================

def train():
	FRAME_DIMS = (30, 30)
	hidden = 100		# hidden layers in model
	lr = .001			# learing rate
	decay_rate = .99
	epsilon = .2		# starting value for exploration

	gamma = .99			# discount factor for reward
	epochs = 1000		# how many sets of batches to go through
	num_actions =  4	# number of different actions that can be taken

	# setup the game
	env = Game(FRAME_DIMS[0], FRAME_DIMS[1])
	# setup the neural net
	net = nn(FRAME_DIMS[0] * FRAME_DIMS[1], num_actions, lr, decay_rate, hidden)

	for g in range(epochs):
		print('Game #{}'.format(g))
		alive = True
		observations, states, loss_logs, rewards = [], [], [], []
		while alive:
			# TODO: SOMETHING WITH FRAME
			# get the difference between the last frame and the current frame
			# set the last frame to the current frame
			frame =  env.board

			# action_prob should be of form [up, down, left, right] for pacman
			# decide if we should explore or if we should take the advice of our network
			action_prob, hidden_state = net.forward(frame)
			explore = float(np.random.randint(1, 100) / 100) <= epsilon
			action = np.argmax(action_prob) if not explore else np.random.randint(len(action_prob))

			# keep track of observations and states for back propagation
			observations.append(frame)
			states.append(hidden_state)

			# keep track of 'losses'. its a bit different for RL as the data isn't labeled
			loss_logs.append(action - action_prob[action])
			
			# get the reward from the environment
			rewards.append(env.move(action))
			# see if the game is over
			alive = env.isOver

		# reduce and normalize rewards based on time
		reduced_rewards = reward_reduction(rewards, gamma)
		reduced_rewards -= np.mean(reduced_rewards)
		reduced_rewards /= np.std(reduced_rewards)

		loss_logs *= reduced_rewards
		for i in range(len(observations)):
			net.backward(observations[i], loss_logs[i])


def play(): 
	pass

if __name__ == '__main__':
	if len(sys.argv) > 1 and '-t' in sys.argv[1]:
		train()
	else:
		play()
