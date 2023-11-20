from Game import Game
from GameConfig import *
from Network import Network

class TrainGame:
    def __init__(self, model_file=None):
        self.gamma = 0.5
        self.alpha = 0.01

        self.network = Network(BOARD_WIDTH, BOARD_HEIGHT, len(SHIPS))
        if not model_file is None:
            self.network.restoreModel(model_file)

        self.game = Game(BOARD_WIDTH, BOARD_HEIGHT, SHIPS, network=self.network)

        self.total_ships_lengths = sum([ship['length'] for ship in self.game.board.ships])

        self.board_size = self.game.board.b_height * self.game.board.b_width

        self.max_train_step = 1000

        self.model_file = model_file

    def rewardsCalculator(self, hit_log, gamma=0.5):
        hit_log_weighted = [(item -
            float(self.total_ships_lengths - sum(hit_log[:index])) / float(self.board_size - index)) *
            (gamma ** index) for index, item in enumerate(hit_log)]
        
    # Dodatna nagrada za pogodak susednih polja
    #   for i in range(len(hit_log)):
    #       if hit_log[i] == 1:  # Ako je trenutni potez pogodak
    #            if i > 0 and hit_log[i-1] == 0:  # Provera levo
    #                hit_log_weighted[i-1] += 0.1
    #            if i < len(hit_log) - 1 and hit_log[i+1] == 0:  # Provera desno
    #                hit_log_weighted[i+1] += 0.1
    #            if i >= self.game.board.board_width and hit_log[i-self.game.board.board_width] == 0:  # Provera gore
    #                hit_log_weighted[i-self.game.board.board_width] += 0.1
    #            if i < self.board_size - self.game.board.board_width and i+self.game.board.board_width < len(hit_log) and hit_log[i+self.game.board.board_width] == 0:
    #                hit_log_weighted[i+self.game.board.board_width] += 0.1

    #            if hit_log[:i].count(1) < len(self.game.board.ships):
    #                ship_length = self.game.board.ships[hit_log[:i].count(1)]['length']
    #                if sum(hit_log[i:i+ship_length]) == ship_length:
    #                    hit_log_weighted[i-1] += 1.0

        # Dodatna nagrada za otkrivanje novih brodova
    #    if hit_log.count(1) < len(self.game.board.ships):
    #        undiscovered_ships = [ship for ship in self.game.board.ships if ship['length'] > 1 and hit_log.count(1) >= hit_log[:ship['length']].count(1)]
    #        for ship in undiscovered_ships:
    #            if hit_log.count(1) >= hit_log[:ship['length']].count(1):
    #                hit_log_weighted[len(hit_log)-ship['length']] += 0.5
        return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]


    def selfPlayOneGame(self):
        all_input_states = []
        all_moves = []
        all_hits = []
        self.game.resetBoard()
        (input_dimensions, move, is_hit) = self.game.takeAMove()
        while not input_dimensions is None:
            all_input_states.append(input_dimensions)
            all_moves.append(move)
            all_hits.append(is_hit)
            (input_dimensions, move, is_hit) = self.game.takeAMove()

        all_discounted_reward = self.rewardsCalculator(all_hits)
        return (all_input_states, all_moves, all_hits, all_discounted_reward)

    def trainWithSelfPlay(self):
        all_num_move = 0
        batch_size = 10
        for i in range(self.max_train_step):
            (all_input_states, all_moves, all_hits, all_discounted_reward) = self.selfPlayOneGame()
            all_num_move += len(all_hits)
            for input_states, moves, discounted_reward in zip(all_input_states, all_moves, all_discounted_reward):
                entropy = self.network.trainStep(input_states, [moves], self.alpha * discounted_reward)
            if i % batch_size == 0 and i != 0:
                print('Game Num: ' + str(i) + ' ' + ' Average moves: ' + str(all_num_move * 1.0/ batch_size * 1.0))
                all_num_move = 0

                if i % (batch_size * 10) == 0:
                    self.network.saveModel(self.model_file)
                    print("SAVE MODEL # {}".format(i))

model_file = './models/mymodel'
train_game = TrainGame(model_file)
train_game.trainWithSelfPlay()