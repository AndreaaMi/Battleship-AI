import numpy as np
import copy
import sys

from Game import Game
from GameConfig import *
from Network import Network

class HumanVSAI:
    def __init__(self, model_file=None):
        self.network = Network(BOARD_WIDTH, BOARD_HEIGHT, len(SHIPS))
        if not model_file is None:
            self.network.restoreModel('./models/mymodel')

        self.ai_game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, network=self.network)
        self.human_game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS)
        

    def playOneGame(self):
        ai_input_dimensions = self.ai_game.board.getInputDimensions()
        human_input_dimensions = self.human_game.board.getInputDimensions()
        cur_step = 0
        while True:

            if ai_input_dimensions is None:
                print('AI wins!')
                sys.exit()
            if human_input_dimensions is None:
                print('You win!')
                sys.exit()

            print('Step: ' + str(cur_step))
            self.printBothBoards()

            ai_available_moves = self.ai_game.board.getNextLocations()
            ai_next_move = self.ai_game.getBestMove(ai_input_dimensions, ai_available_moves)
            (ai_input_dimensions, _, _) = self.ai_game.takeAMove(ai_next_move)

            human_available_moves = self.human_game.board.getNextLocations()
            human_next_move = self.getHumanMoveInput()
            if human_next_move == None:
                human_next_move = self.human_game.getRandomMove(human_available_moves)
            (human_input_dimensions, _, _) = self.human_game.takeAMove(human_next_move)

            cur_step += 1


    def getHumanMoveInput(self):
        while True:
            human_move = input("Enter the coordinates of the next move in the format 'x,y' or you can enter 'random' if you want a random move: ")
            if human_move == '':
                return None
            if human_move.lower() == 'random':
                return self.human_game.getRandomMove(self.human_game.board.getNextLocations())
            xy = human_move.split(',')
            if len(xy) != 2:
                print("Invalid format! Please enter the coordinates in the format 'x,y'.")
                continue
            x = xy[0].strip()
            y = xy[1].strip()
            if not x.isdigit() or not y.isdigit():
                print("Invalid format! Please enter numeric values for coordinates.")
                continue
            x = int(x)
            y = int(y)
            if x >= self.human_game.board.b_height or x < 0 or y >= self.human_game.board.b_width or y < 0:
                print("Invalid coordinates! Please enter valid coordinates within the board range.")
                continue
            location = x * self.human_game.board.b_width + y
            if self.human_game.board.bomb_locations[location] != 1:
                print("Invalid move! The specified location is not available for bombing.")
                continue
            return location
    
    def printBothBoards(self):
        ai_board_printer = self.ai_game.board.getViewState()
        human_board_printer = self.human_game.board.getViewState()
        spaces = '                      '
        for i in range(len(ai_board_printer)):
            print(ai_board_printer[i] + spaces + human_board_printer[i])

gamer = HumanVSAI('./models/mymodel')
gamer.playOneGame()
