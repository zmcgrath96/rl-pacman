import sys
import numpy as np
from utils import process_frame
from nn import nn
from game import Game
import time

# ===============================================================================================
#							Helper functions 
# ===============================================================================================
def reward_reduction(rewards, gamma):
	reduced = [0 for _ in range(len(rewards))]
	running_add = 0
	# increase the rewards for things that happened towards the end of the game
	for t in range(len(rewards)):
		running_add =  np.multiply(running_add, gamma) + rewards[t]
		reduced[t] = running_add

	return np.asarray(reduced)

def to_one_hot(n, n_classes, val=1):
	ret = [0] * n_classes
	ret[n] = val
	return np.array(ret)

# ===============================================================================================
#							END Helper functions 
# ===============================================================================================

def train():
	FRAME_DIMS = (10, 10)
	hidden = 100		# hidden layers in model
	lr = .0001			# learing rate
	decay_rate = .99
	epsilon = .2		# starting value for exploration
	reduce_epsilon = .999992

	gamma = .99			# discount factor for reward
	epochs = 400000		# how many sets of batches to go through
	num_actions =  4	# number of different actions that can be taken

	# setup the neural net
	net = nn(FRAME_DIMS[0] * FRAME_DIMS[1], num_actions, lr, decay_rate, hidden)

	max_moves = 100000

	win = 0
	loss = 0

	total_wins = 0
	total_loss = 0

	decades = []
	decade = 0
	g_in_d = 0

	for g in range(1, epochs + 1):
		# setup the game
		env = Game(FRAME_DIMS[0], FRAME_DIMS[1])
		alive = True
		observations, states, loss_logs, rewards = [], [], [], []
		num_moves = 0
		p = True
		diff_moves = [0] * num_actions
		while alive and max_moves > num_moves:
			
			frame =  env.board

			# action_prob should be of form [up, down, left, right] for pacman
			# decide if we should explore or if we should take the advice of our network
			action_prob, hidden_state = net.forward(frame)
			explore = float(np.random.randint(1, 100) / 100) <= epsilon
			action = np.argmax(action_prob) if not explore else np.random.randint(len(action_prob))
			diff_moves[action] += 1
			action = to_one_hot(action, len(action_prob))
			if not p:
				print(action)
				print(action_prob)
				p = True

			# keep track of observations and states for back propagation
			observations.append(frame)
			states.append(hidden_state)

			# keep track of 'losses'. its a bit different for RL as the data isn't labeled
			loss_logs.append(action - action_prob)
			
			# get the reward from the environment
			a = int(np.argmax(action))
			r = to_one_hot(a, len(action), env.move(a))
			rewards.append(r)
			# see if the game is over
			alive = not env.isOver and not env.died
			num_moves += 1

			if env.died:
				loss += 1
				total_loss += 1
			elif env.isOver:
				win += 1
				total_wins += 1
			# print('move: {}'.format(num_moves))
			# env.renderBoard()
			# time.sleep(.1)

		# reduce and normalize rewards based on time
		reduced_rewards = reward_reduction(rewards, gamma)
		reduced_rewards -= np.mean(reduced_rewards)
		reduced_rewards /= np.std(reduced_rewards)

		loss_logs = np.array(loss_logs)
		loss_logs = np.multiply(loss_logs, reduced_rewards)
		for i in range(len(observations)):
			net.backward(observations[i], loss_logs[i])

		epsilon = epsilon if epsilon <= .05 else epsilon * reduce_epsilon
		g_in_d += 1
		
		print('Game {} \t Overall win rate: {:.2f} \t Decade win rate: {:.2f} \t Epsilon: {:.2f} \r'.format(g, round(float(total_wins / g), 4) * 100, round(float(win / g_in_d), 4) * 100  ,epsilon))

		if g % 10000 == 0:
			decades.append('Decade: {} \t Wins {}, \t losses: {}'.format(decade, win, loss))
			win = 0
			loss = 0
			decade += 1
			g_in_d = 0

	for d in decades:
		print(d)
	print('TOTAL: \t won {} games, lost {} games'.format(win, loss))

def play(): 
	pass

if __name__ == '__main__':
	if len(sys.argv) > 1 and '-t' in sys.argv[1]:
		train()
	else:
		play()
