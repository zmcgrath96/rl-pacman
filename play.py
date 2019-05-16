import sys
import numpy as np
from game import Game
import time
import pickle

NUM_ACTIONS = 4
MIN_ALPHA = 0.02
NUM_EPISODES = 600000
SLIDING_WINDOW = 1000
FRAME_SIZE = (10, 10)

MAX_EPISODE_STEPS = 100

PICKLE_FILE = 'qtable.pickle'

alphas = np.linspace(1.0, MIN_ALPHA, NUM_EPISODES)
gamma = 1.0

def train():
	eps = 0.3
	wins = [0] * SLIDING_WINDOW
	winrate = 0.0
	episode = 0
	totalRewardArr = [0] * SLIDING_WINDOW
	avgTotalReward = 0.0
	qTable = dict()
	while episode < NUM_EPISODES and winrate < 0.999:

		env = Game(FRAME_SIZE[0], FRAME_SIZE[1])

		state = env.state()
		totalReward = 0
		alpha = alphas[episode]

		for _ in range(MAX_EPISODE_STEPS):
			action = chooseAction(qTable, state, eps)
			reward = env.move(action)
			nextState = env.state()
			done = env.isOver
			totalReward += reward

			getQ(qTable, state)[action] = getQ(qTable, state, action) + alpha * (reward + gamma * np.max(getQ(qTable, nextState)) - getQ(qTable, state, action))
			state = nextState
			if done:
				break
		if env.isOver and not env.isDead:
			wins.append(1)
		else:
			wins.append(0)
		wins.pop(0)
		totalRewardArr.append(totalReward)
		totalRewardArr.pop(0)
		episode += 1
		if eps > 0.01 and 1.0 - winrate < eps:
			eps -= eps * 0.0001
		winrate = sum(wins) / SLIDING_WINDOW
		avgTotalReward = sum(totalRewardArr) / SLIDING_WINDOW
		print("Episode {}: \t total reward avg -> {:.2f} \t win rate -> {:.2f} \t qtable size -> {} \t eps -> {:.2f}".format(episode + 1, avgTotalReward, winrate, len(qTable), eps), end='\r')
	pickle.dump(qTable, open(PICKLE_FILE, 'wb'))

def getQ(table, state, action=None):
	if state not in table:
		table[state] = np.zeros(NUM_ACTIONS)
	if action is None:
		return table[state]
	
	return table[state][action]

def chooseAction(table, state, eps):
	if np.random.uniform(0,1) < eps:
		return np.random.choice(NUM_ACTIONS)
	return np.argmax(getQ(table, state))

def play():
	
	qTable = pickle.load(open(PICKLE_FILE, 'rb'))
	env = Game(FRAME_SIZE[0], FRAME_SIZE[1])
	print(env.renderBoard())
	while not env.isOver:
		action = chooseAction(qTable, env.state(), 0.01)
		env.move(action)
		print(env.renderBoard())
		time.sleep(0.1)

def test():
	qTable = pickle.load(open(PICKLE_FILE, 'rb'))
	wins = 0
	moves = []
	for _ in range(1000):
		env = Game(FRAME_SIZE[0], FRAME_SIZE[1])
		move = 0
		while not env.isOver:
			action = chooseAction(qTable, env.state(), 0.01)
			env.move(action)
			move += 1
		if not env.isDead:
			wins += 1
			moves.append(move)
	print('Win rate: {:.2f} \t avg moves: {:.2f}'.format(wins/1000, sum(moves)/len(moves)))
		

if __name__ == '__main__':
	if len(sys.argv) > 1 and '-train' in sys.argv[1]:
		train()
	elif len(sys.argv) > 1 and '-test' in sys.argv[1]:
		test()
	else:
		play()
