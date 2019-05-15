import sys
import numpy as np
from game import Game
import time
import pickle

NUM_ACTIONS = 4
MIN_ALPHA = 0.02
NUM_EPISODES = 1000000
SLIDING_WINDOW = 1000
FRAME_SIZE = (10, 10)

MAX_EPISODE_STEPS = 100

PICKLE_FILE = 'qtable.pickle'

alphas = np.linspace(1.0, MIN_ALPHA, NUM_EPISODES)
gamma = 1.0
eps = 0.2

qTable = dict()

def train():	
	wins = [0] * SLIDING_WINDOW
	winrate = 0.0
	episode = 0
	totalRewardArr = [0] * SLIDING_WINDOW
	avgTotalReward = 0.0
	
	while episode < NUM_EPISODES and (avgTotalReward < 100 or winrate < 1.0):

		env = Game(FRAME_SIZE[0], FRAME_SIZE[1])

		state = env.state()
		totalReward = 0
		alpha = alphas[episode]

		for _ in range(MAX_EPISODE_STEPS):
			action = chooseAction(state)
			reward = env.move(action)
			nextState = env.state()
			done = env.isOver or env.isDead
			totalReward += reward

			getQ(state)[action] = getQ(state, action) + alpha * (reward + gamma * np.max(getQ(nextState)) - getQ(state, action))
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
		winrate = sum(wins) / SLIDING_WINDOW
		avgTotalReward = sum(totalRewardArr) / SLIDING_WINDOW
		print("Episode {}: \t total reward avg -> {:.2f} \t win rate -> {:.2f} \t qtable size: {}".format(episode + 1, avgTotalReward, winrate, len(qTable)))
	pickle.dump(qTable, open(PICKLE_FILE, 'wb'))

def getQ(state, action=None):
	if state not in qTable:
		qTable[state] = np.zeros(NUM_ACTIONS)
	if action is None:
		return qTable[state]
	
	return qTable[state][action]

def chooseAction(state):
	if np.random.uniform(0,1) < eps:
		return np.random.choice(NUM_ACTIONS)
	return np.argmax(getQ(state))

def play():
	env = Game(FRAME_SIZE[0], FRAME_SIZE[1])
	qTable = pickle.load(open(PICKLE_FILE, 'rb'))
	
	while not env.isOver:
		print(env.renderBoard())
		action = chooseAction(env.state())
		env.move(action)
		time.sleep(0.1)

if __name__ == '__main__':
	if len(sys.argv) > 1 and '-t' in sys.argv[1]:
		train()
	else:
		play()
