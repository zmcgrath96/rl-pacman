import numpy as np

EMPTY = 0
WALL = 1
PLAYER = 2
EXIT = 3
KEY = 4

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

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
        while self.keyPos is self.playerPos:
            self.keyPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        self.board[self.keyPos[0], self.keyPos[1]] = KEY
        self.exitPos = self.playerPos
        while self.exitPos is self.playerPos or self.exitPos is self.keyPos:
            self.exitPos = (np.random.randint(1, height - 1), np.random.randint(1, width - 1))
        self.board[self.exitPos[0], self.exitPos[1]] = EXIT
        self.isOver = False


    
    def renderBoard(self):
        shape = self.board.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                print(tileDict[self.board[i,j]], end='')
            print('')

    def move(self, direction):
        pass


game = Game(10, 20)
game.renderBoard()