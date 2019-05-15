import numpy as np

EMPTY = 0.0
WALL = 0.1
PLAYER = 0.2
EXIT = 0.3
KEY = 0.4

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

NEG_REWARD = -1
ILLEGAL_REWARD = -100
KEY_REWARD = 70
EXIT_REWARD = 100

tileDict = {EMPTY: ' ', WALL: 'W', PLAYER: 'P', EXIT: 'E', KEY: 'K'}

class Game:
    def __init__(self, height, width):
        self.board = np.full((height, width), EMPTY)
        self.board[0, :] = WALL
        self.board[-1, :] = WALL
        self.board[:, 0] = WALL
        self.board[:, -1] = WALL
        self.keyPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        self.board[self.keyPos[0], self.keyPos[1]] = KEY
        self.exitPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        while self.exitPos == self.keyPos:
            self.exitPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        self.board[self.exitPos[0], self.exitPos[1]] = EXIT
        self.playerPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        while self.playerPos == self.keyPos or self.playerPos == self.exitPos:
            self.playerPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        self.board[self.playerPos[0], self.playerPos[1]] = PLAYER
        self.isOver = False
        self.hasKey = False
        self.isDead = False


    
    def renderBoard(self):
        if self.isOver:
            print('You have escpaed!')
        shape = self.board.shape
        board = ''
        for i in range(shape[0]):
            for j in range(shape[1]):
                board += tileDict[self.board[i,j]]
            board += '\n'
        return board
    
    def state(self):
        shape = self.board.shape
        board = ''
        for i in range(shape[0]):
            for j in range(shape[1]):
                board += tileDict[self.board[i,j]]
        return board

    def move(self, direction):
        
        newPos = self.getNewPos(direction)
        if self.isWall(newPos):
            self.isDead = True
            self.isOver = True
            reward = ILLEGAL_REWARD
        elif self.isValidMove(newPos):
            reward = self.deterimineReward(newPos)
            self.updatePlayerPos(newPos)
        else:
            reward = ILLEGAL_REWARD
        return reward

    def getNewPos(self, direction):
        if direction == UP:
            return (self.playerPos[0] - 1, self.playerPos[1])
        elif direction == DOWN:
            return (self.playerPos[0]  + 1 , self.playerPos[1])
        elif direction == LEFT:
            return (self.playerPos[0], self.playerPos[1] - 1)
        elif direction == RIGHT:
            return (self.playerPos[0], self.playerPos[1] + 1)
    
    def updatePlayerPos(self, pos):
        self.board[self.playerPos[0], self.playerPos[1]] = EMPTY
        self.playerPos = pos
        self.board[self.playerPos[0], self.playerPos[1]] = PLAYER        
    
    def deterimineReward(self, pos):
        if not self.hasKey and self.board[pos[0], pos[1]] == KEY:
            self.hasKey = True
            return KEY_REWARD
        
        if self.hasKey and  self.board[pos[0], pos[1]] == EXIT:
            self.isOver = True
            return EXIT_REWARD
        
        return NEG_REWARD

    def isValidMove(self, newPos):
        pos = self.board[newPos[0], newPos[1]]
        if pos == PLAYER or (not self.hasKey and pos == EXIT):
            return False
        return True
    
    def isWall(self, newPos):
        pos = self.board[newPos[0], newPos[1]]
        if pos == WALL:
            return True
        return False