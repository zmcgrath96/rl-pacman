# Treasure Hunters Inc
Repository for EECS 738 Project Reinforcement Learning

#### Authors
Zachary McGrath, Kevin Ray (github.com/kray10)

## Installation

Clone the repo
```
%> git clone https://github.com/zmcgrath96/rl-pacman
```

Install dependencies
```
$> pip3 install -r requirements.txt
```
## The Game
We created a simple game for our Q-learning algorithm to train on. The board is a 10 x 10 2D array where the perimeter is contains lava denoted as an 'L'. There is also a 'P' for the player position, a 'K' for the key position, and an 'E' for the exit postion. The player, key, and exit positions are all randomly generated for each game. The goal of the game is to move to the key location and then move to the exit postion without touching the lava. The game allows four inputs: up, down, left, and right.

```
  LLLLLLLLLL
  L        L
  L        L
  L  P     L
  L        L
  L        L
  LE       L
  L        L
  L      K L
  LLLLLLLLLL
```

## Running
### Training
```
%> python3 play.py -train
```
Training is done using the Q-learning method. A dictionary is implemented to store all of the game states that the training has seen thus far. The each game state is the game board rendered as a string. This always for easy hashing in the dictionary. Each dictionary entry contains an array of length four that respresents the expected reward for each game action in that state. These values are adjusted as the training makes different moves from that state.

The training also consideres an exploration rate, know as epsilon, to dictate how often the training will explore, ie make a random move rather than the move that will result in the highest reward. This rate is originally set at 30%. Since the lava results in many instant losses due to random moves, it is difficult for the training's win rate to exceed 1.0 - epsilon by more than the 0.25 * epsilon since there are 4 random actions that can taken. As a result, we begin to decrease epsilon when the 1.0 - win rate exceeds epsilon. Decreasing epsilon too much can cause the game to get stuck in loop if all states have not been discovered as you can have two states where the highest reward cycle into each other. Reducing epsilon can also result in new states not being discovered quickly enoughing which can slow down training. For these two reasons epsilon never decreases below 0.01.

### Testing
```
%> python3 play.py -test
```
This will load the trained q table from qTable.pickle and then play 1,000 games. The epsilon for this testing is set to 0.01 to make sure that the game does not get stuck. The testing will output the win rate and the average number of moves it took to win the game. For the training data provided, the win rate is typically 97% - 98% and the average number of move are 15 - 18.

### Playing
```
%> python3 play.py
```
This will also load from the training data, but it only plays a single game. The board is printed to the screen after each move with a 0.1 second delay btween moves.

## Results
Q-learning seemed to be very effective for this simple of a game. Originally, the game did not have randomly generated key and exit positions. This resulted in much smaller training times and state spaces. For that version, the training typically resulted in just over 1,000 states being discovered during training. Adding randomized positions for all object increased that state space to over 250,000. 

In the branch qlearing-enemy, we also toyed with the idea of adding an enemy that moved around the board. This resulted in two main problems:
1. The enemie's movement was randomly generated. This made the reward for each action flucuate heavily. We attempted to counter this by having the enemy move toawrds the player with a predefined set of rules.
2. Adding the enemy blew the state space to over 15,000,000. This increased training time drastically. We tried to fix this by limiting the area that the training looked at. Instead of considering the whole board, it considered only the 5x5 grid surrounding the player. While this would reduce the state space, we knew it would also make it harder to find the key and exit since they would not always be visible to the training.

After implementing these changes, we found that the game became to hard for the training to win. This was most likely due to the reduced vision that the training had. There were too many states where the training did not have any idea where the key or exit were to determine which action would produce a higher reward. We feel this resulted in equally likely rewards for all actions making the movemeny essentially random. 

