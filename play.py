import sys
import numpy as np
from game import Game
import time

NUM_ACTIONS = 4
MIN_ALPHA = 0.02
NUM_EPISODES = 10000

MAX_EPISODE_STEPS = 100

alphas = np.linspace(1.0, MIN_ALPHA, NUM_EPISODES)
gamma = 1.0
eps = 0.2

qTable = dict()

def train():	

	for episode in range(NUM_EPISODES):

		env = Game(10, 10)

		state = env.state()
		totalReward = 0
		alpha = alphas[episode]

		for _ in range(MAX_EPISODE_STEPS):
			action = chooseAction(state)
			# print(env.renderBoard())
			reward = env.move(action)
			nextState = env.state()
			done = env.isOver
			totalReward += reward

			getQ(state)[action] = getQ(state, action) + alpha * (reward + gamma * np.max(getQ(nextState)) - getQ(state, action))
			state = nextState
			# print(getQ(state))
			# time.sleep(5)
			if done:
				break
		print("Episode {}: total reward -> {}".format(episode + 1, totalReward))

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

if __name__ == '__main__':
	if len(sys.argv) > 1 and '-t' in sys.argv[1]:
		train()
