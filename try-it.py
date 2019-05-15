""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
from game import Game
from utils import *

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
actions = 4

# model initialization
D = 10 * 10 # input dimensionality: 80x80 grid

model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
model['W2'] = np.random.randn(H, actions) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(h, model['W2'])
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp)
  dh = np.dot(epdlogp, model['W2'].T)
  dh[dh <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = Game(10, 10)
observation = env.board
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
epsilon = .1
wins = 0
max_games = 10000000
desired_win_rate = 30.0
win_rate = 0.0

rates = [0] * 1000

while episode_number < max_games or win_rate > desired_win_rate:

  # preprocess the observation, set input to network to be difference image
  x = observation.flatten()

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = np.argmax(aprob) if np.random.randint(0, 100) / 100 > epsilon else np.random.randint(0,4)

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  a_hot = [0] * actions
  a_hot[action] = 1
  dlogps.append(a_hot - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation = env.board
  reward = env.move(action)
  done = env.isOver or env.died
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  if done: # an episode finished
    win = not env.died
    if win: 
      wins += 1
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: 
      grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k in model:
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(model[k]) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    #print('episode reward total was {}. running mean: {}'.format(reward_sum, running_reward))
    win_rate = (wins / episode_number) * 100
    rates.append(1 if win else 0)
    rates.pop(0)
    sliding_rate = float(sum(rates)/len(rates)) * 100
    print('wins: {} \t games: {} \t win rate: {:.2f} \t sliding rate: {:.2f}'.format(wins, episode_number, win_rate, sliding_rate))
    reward_sum = 0
    env = Game(10, 10)


with open('results.txt', 'w') as o:
  o.write('win rate: {}'.format(win_rate))
  o.wirte('wins: {}'.format(wins))
  o.write('games: {}'.format(episode_number))