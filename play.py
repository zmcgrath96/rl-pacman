import sys
import numpy as np
from utils import process_frame
from nn import NeuralNetwork
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
	epsilon = .2		# starting value for exploration
	reduce_epsilon = .999992

	epochs = 400000		# how many sets of batches to go through
	num_actions =  4	# number of different actions that can be taken

	# setup the neural net
	net = NeuralNetwork(FRAME_DIMS[0] * FRAME_DIMS[1], num_actions, hidden)

	max_moves = 1000

	win = 0
	loss = 0

	total_wins = 0
	total_loss = 0

	last_100_moves = [0] * 100
	game_index = 0

	decades = []
	decade = 0
	g_in_d = 0

	JUST_SAVE_STAT = True

	for g in range(1, epochs + 1):
		# setup the game
		env = Game(FRAME_DIMS[0], FRAME_DIMS[1])
		alive = True
		num_moves = 0
		diff_moves = [0] * num_actions
		while alive and max_moves > num_moves:
			
			frame =  np.reshape(env.board, (FRAME_DIMS[0] * FRAME_DIMS[1], 1))

			# action_prob should be of form [up, down, left, right] for pacman
			# decide if we should explore or if we should take the advice of our network
			hidden_state, action_prob = net.forward(frame)
			explore = float(np.random.randint(1, 100) / 100) <= epsilon
			action = np.argmax(action_prob) if not explore else np.random.randint(len(action_prob))
			diff_moves[action] += 1
			# print(action)
			reward = env.move(action)
			reward_arr = np.reshape(to_one_hot(action, num_actions, reward), (num_actions, 1))
			net.backward(hidden_state, action_prob, frame, reward_arr)

			# see if the game is over
			alive = not env.isOver and not env.died
			num_moves += 1

			if env.died:
				loss += 1
				total_loss += 1
			elif env.isOver:
				win += 1
				total_wins += 1
			
			# env.renderBoard()
			# time.sleep(0.1)
		
		last_100_moves.append(num_moves)
		last_100_moves.pop(0)
		if game_index >= 99:
			game_index = 0
		else:
			game_index += 1
		
		average_moves = np.sum(np.array(last_100_moves)) / 100.0

		epsilon = epsilon if epsilon <= .05 else epsilon * reduce_epsilon
		g_in_d += 1
		
		print('Game {} \t Overall win rate: {:.2f} \t Decade win rate: {:.2f} \t Epsilon: {:.2f} \t Average Moves: {:.2f}\t\r'.format(g, round(float(total_wins / g), 4) * 100, round(float(win / g_in_d), 4) * 100  ,epsilon, average_moves), end='')

		if g % 10000 == 0:
			decades.append('Decade: {} \t Wins {}, \t losses: {}'.format(decade, win, loss))
			win = 0
			loss = 0
			decade += 1
			g_in_d = 0

	for d in decades:
		print(d)
	print('TOTAL: \t won {} games, lost {} games'.format(win, loss))

	if JUST_SAVE_STAT:
		with open('stat.txt', 'w') as o:
			for d in decades:
				o.write(d)
			o.write('TOTAL: \t won {} games, lost {} games'.format(win, loss))

def play(): 
	pass

if __name__ == '__main__':
	if len(sys.argv) > 1 and '-t' in sys.argv[1]:
		train()
	else:
		play()
