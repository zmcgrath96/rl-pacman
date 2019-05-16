import numpy as np
import math

EMPTY = 0
LAVA = 1
PLAYER = 2
EXIT = 3
KEY = 4
ENEMY = 5

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

NEG_REWARD = -1
ILLEGAL_REWARD = -100
KEY_REWARD = 700
EXIT_REWARD = 1000
KILLED_REWARD = -1000

tileDict = {EMPTY: ' ', LAVA: 'L', PLAYER: 'P', EXIT: 'X', KEY: 'K', ENEMY: 'E'}

class Game:
    def __init__(self, height, width):
        
        self.board = np.full((height + 2, width + 2), EMPTY)
        self.board[0:2, :] = LAVA
        self.board[-2:, :] = LAVA
        self.board[:, 0:2] = LAVA
        self.board[:, -2:] = LAVA
        self.keyPos = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
        self.board[self.keyPos[0], self.keyPos[1]] = KEY
        self.exitPos = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
        while self.exitPos == self.keyPos:
            self.exitPos = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
        self.board[self.exitPos[0], self.exitPos[1]] = EXIT
        self.playerPos = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
        while self.playerPos == self.keyPos or self.playerPos == self.exitPos:
            self.playerPos = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
        self.enemyPos = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
        while self.enemyPos == self.keyPos or self.enemyPos == self.exitPos or self.enemyPos == self.playerPos:
            self.enemyPos = (np.random.randint(2, height - 2), np.random.randint(2, width - 2))
        self.board[self.playerPos[0], self.playerPos[1]] = PLAYER
        self.board[self.enemyPos[0], self.enemyPos[1]] = ENEMY
        self.isOver = False
        self.hasKey = False
        self.isDead = False
        self.numEnemies  = math.ceil(height * width / 100)
        self.enemyMove = 0

    
    def renderBoard(self):
        if self.isOver and not self.isDead:
            return 'You have escpaed!'
        
        if self.isDead:
            return 'You have died!'

        shape = self.board.shape
        board = ''
        for i in range(shape[0]):
            for j in range(shape[1]):
                board += tileDict[self.board[i,j]]
            board += '\n'
        return board
    
    def state(self):
        board = ''
        for i in range(self.playerPos[0] - 2, self.playerPos[0] + 3):
            for j in range(self.playerPos[1] - 2, self.playerPos[1] + 3):
                board += tileDict[self.board[i,j]]
        return board

    def move(self, direction):
        # player movement
        newPos = self.getNewPos(direction)
        self.enemyMove += 1
        #enemy movement 
        if self.enemyMove > 2:
            self.enemyMove = 0
            de_x = abs(self.enemyPos[0] - self.playerPos[0])
            de_y = abs(self.enemyPos[1] - self.playerPos[1])

            move = [0, 0]

            if self.playerPos[0] >= self.enemyPos[0]:
                move[0] = 1
            elif self.playerPos[0] <= self.enemyPos[0]:
                move[0] = -1
            
            if self.playerPos[1] >= self.enemyPos[1]:
                move[1] = 1
            elif self.playerPos[1] <= self.enemyPos[1]:
                move[1] = -1

            # move in x direction
            if de_x > de_y:
                pos = (move[0] + self.enemyPos[0], self.enemyPos[1])
                if self.isValidMove(pos, player='E'):
                    self.updateEnemyPos(pos)
            # move in y direction
            else:
                pos = (self.enemyPos[0], move[1] + self.enemyPos[1])
                if self.isValidMove(pos, player='E'):
                    self.updateEnemyPos(pos)
                
        self.enemyMove = not self.enemyMove

        # see if enemy got the player
        if self.enemyPos and self.enemyPos == self.playerPos:
            reward = KILLED_REWARD
            self.isDead = True
            self.isOver = True

        # see if the player is running into the enemy
        elif self.isEnemy(newPos):
            reward = KILLED_REWARD
            self.isDead = True
            self.isOver = True

        # see if the player is running into the LAVA
        elif self.isLAVA(newPos):
            self.isDead = True
            self.isOver = True
            reward = ILLEGAL_REWARD

        # see if the player is making a valid choice
        elif self.isValidMove(newPos):
            reward = self.deterimineReward(newPos)
            self.updatePlayerPos(newPos)

        else:
            reward = ILLEGAL_REWARD

        return reward

    def getNewPos(self, direction, player='P'):
        pos = self.playerPos
        if self.enemyPos and 'E' in player:
            pos = self.enemyPos

        if direction == UP:
            return (pos[0] - 1, pos[1])
        elif direction == DOWN:
            return (pos[0]  + 1 , pos[1])
        elif direction == LEFT:
            return (pos[0], pos[1] - 1)
        elif direction == RIGHT:
            return (pos[0], pos[1] + 1)
    
    def updatePlayerPos(self, pos):
        self.board[self.playerPos[0], self.playerPos[1]] = EMPTY
        self.playerPos = pos
        self.board[self.playerPos[0], self.playerPos[1]] = PLAYER     

    def updateEnemyPos(self, pos):
        self.board[self.enemyPos[0], self.enemyPos[1]] = EMPTY
        self.enemyPos = pos
        self.board[self.enemyPos[0], self.enemyPos[1]] = ENEMY       
    
    def deterimineReward(self, pos):
        if not self.hasKey and self.board[pos[0], pos[1]] == KEY:
            self.hasKey = True
            return KEY_REWARD
        
        if self.hasKey and self.board[pos[0], pos[1]] == EXIT:
            self.isOver = True
            return EXIT_REWARD
        
        return NEG_REWARD

    def isValidMove(self, newPos, player='P'):
        pos = self.board[newPos[0], newPos[1]]
        if 'E' in player:
            if pos == ENEMY or pos == LAVA or pos == EXIT or pos == KEY:
                return False
            return True
        else:
            if pos == PLAYER or (not self.hasKey and pos == EXIT):
                return False
            return True
    
    def isLAVA(self, newPos):
        pos = self.board[newPos[0], newPos[1]]
        if pos == LAVA:
            return True
        return False

    def isEnemy(self, newPos):
        pos = self.board[newPos[0], newPos[1]]
        if pos == ENEMY:
            return True
        return False