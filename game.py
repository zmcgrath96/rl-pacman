import numpy as np

EMPTY = 0
WALL = 100
PLAYER = 200
EXIT = 300
KEY = 400

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

NEG_REWARD = 0
ILLEGAL_REWARD = -1
KEY_REWARD = 1
EXIT_REWARD = 1

tileDict = {EMPTY: ' ', WALL: 'W', PLAYER: 'P', EXIT: 'E', KEY: 'K'}

class Game:
    def __init__(self, height, width):
        self.board = np.full((height, width), EMPTY)
        self.board[0, :] = WALL
        self.board[-1, :] = WALL
        self.board[:, 0] = WALL
        self.board[:, -1] = WALL
        self.playerPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        self.board[self.playerPos[0], self.playerPos[1]] = PLAYER
        self.keyPos = self.playerPos
        while self.keyPos == self.playerPos:
            self.keyPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        self.board[self.keyPos[0], self.keyPos[1]] = KEY
        self.exitPos = self.playerPos
        while self.exitPos == self.playerPos or self.exitPos == self.keyPos:
            self.exitPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        self.board[self.exitPos[0], self.exitPos[1]] = EXIT
        self.isOver = False
        self.hasKey = False
        self.died = False


    
    def renderBoard(self):
        if self.isOver:
            print('You have escpaed!')
        shape = self.board.shape
        board = ''
        for i in range(shape[0]):
            for j in range(shape[1]):
                board += tileDict[self.board[i,j]]
            board += '\n'
        print(board)

    def move(self, direction):
        
        newPos = self.getNewPos(direction)

        if self.isWall(newPos):
            reward = ILLEGAL_REWARD
            self.isOver = True
            self.died = True

        elif self.isValidMove(newPos):
            reward = self.deterimineReward(newPos)
            self.updatePlayerPos(newPos)
        else:
            reward = NEG_REWARD
        return reward

    def getNewPos(self, direction):

        newPos = self.playerPos

        if direction is UP:
            newPos = (newPos[0], newPos[1] + 1)
        elif direction is DOWN:
            newPos = (newPos[0], newPos[1] - 1)
        elif direction is LEFT:
            newPos = (newPos[0] - 1, newPos[1])
        elif direction is RIGHT:
            newPos = (newPos[0] + 1, newPos[1])

        return newPos
    
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
        if pos == WALL or pos == PLAYER or (not self.hasKey and pos == EXIT):
            return False
        return True

    def isWall(self, newPos):
        pos = self.board[newPos[0], newPos[1]]
        if pos == WALL:
            return True
        return False