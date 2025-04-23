import random
import copy
from collections import Counter
import matplotlib.pyplot as plt

class Game2048:
    def __init__(self):
        self.board = [[0]*4 for _ in range(4)]
        self.game_over = False
        self.spawn_number_tile()
        self.spawn_number_tile()

    def spawn_number_tile(self):
        empty_tiles = [(row, column) for row in range(4) for column in range(4) if self.board[row][column] == 0]
        if not empty_tiles:
            return
        row, column = random.choice(empty_tiles)
        self.board[row][column] = random.choices([2, 4], weights=[0.9, 0.1])[0]

    def clean_up(self, row):
        new_row = [num for num in row if num != 0]
        new_row += [0] * (4 - len(new_row))
        return new_row

    def merge(self, row):
        for i in range(3):
            if row[i] != 0 and row[i] == row[i+1]:
                row[i] *= 2
                row[i+1] = 0
        return row

    def final_merge(self, row):
        return self.clean_up(self.merge(self.clean_up(row)))

    def move_left(self):
        self.board = [self.final_merge(row) for row in self.board]

    def move_right(self):
        new_board = []

        # final merge but on a reversed board (regular merge only works on left side)
        for row in self.board:
            reverse = row[::-1]
            final_row = self.final_merge(reverse)
            new_board.append(final_row[::-1])
        self.board = new_board

    def move_up(self):
        # transpose board (switch rows with columns)
        transpose = [[self.board[row][column] for row in range(4)] for column in range(4)]
        merged = [self.final_merge(row) for row in transpose]
        # tranpose board back
        self.board = [[merged[c][r] for c in range(4)] for r in range(4)]

    def move_down(self):
        # transpose board (switch rows with columns), and flip the board
        transpose = [[self.board[r][c] for r in range(4)][::-1] for c in range(4)]
        merged = [self.final_merge(row) for row in transpose]
        # flip the board back and then transpose it back
        flipped = [row[::-1] for row in merged]
        self.board = [[flipped[c][r] for c in range(4)] for r in range(4)]

    def move(self, direction):
        old_board = copy.deepcopy(self.board)
        if direction == 'up':
            self.move_up()
        elif direction == 'down':
            self.move_down()
        elif direction == 'left':
            self.move_left()
        elif direction == 'right':
            self.move_right()

        if self.board != old_board:
            self.spawn_number_tile()
        elif not self.can_move():
            self.game_over = True

    def can_move(self):
        for row in range(4):
            for column in range(4):
                if self.board[row][column] == 0:
                    return True
                if column < 3 and self.board[row][column] == self.board[row][column + 1]:
                    return True
                if row < 3 and self.board[row][column] == self.board[row + 1][column]:
                    return True
        return False

    def print_board(self):
        print("\n")
        for row in self.board:
            print("".join(f"{num or '-':>5}" for num in row))
        print("\n")


class Expectimax:
    def __init__(self, weights):
        self.weights = weights  # (corner, merge, empty, full columns/rows)

    def is_max_tile_in_corner(self, board):
        max_tile = max(max(row) for row in board)
        return max_tile in [board[0][0], board[0][3], board[3][0], board[3][3]]

    def count_potential_merges(self, board):
        potential_merges = 0

        for row in board:
            row = [x for x in row if x != 0]
            for i in range(len(row) - 1):
                if row[i] == row[i+1]:
                    potential_merges += 1

        for column in zip(*board):
            column = [x for x in column if x != 0]
            for i in range(len(column) - 1):
                if column[i] == column[i+1]:
                    potential_merges += 1

        return potential_merges

    def count_empty_tiles(self, board):
        return sum(cell == 0 for row in board for cell in row)

    def non_full_lines(self, board):
        rows_with_space = sum(1 for row in board if any(cell == 0 for cell in row))
        cols_with_space = sum(1 for col in zip(*board) if any(cell == 0 for cell in col))
        return rows_with_space + cols_with_space

    def evaluation_function(self, board):
        w_corner, w_merge, w_empty, w_non_full = self.weights

        score = 0
        if self.is_max_tile_in_corner(board):
            score += w_corner
        score += self.count_potential_merges(board) * w_merge
        score += self.count_empty_tiles(board) * w_empty
        score += self.non_full_lines(board) * w_non_full

        return score

    def expectimax(self, game, depth, is_max_turn):
        if depth == 0 or game.game_over:
            return self.evaluation_function(game.board)

        # max over all possible moves
        if is_max_turn:
            best = float('-inf')
            for move in ['up', 'down', 'left', 'right']:
                new_game = copy.deepcopy(game)
                new_game.move(move)
                if new_game.board != game.board:
                    score = self.expectimax(new_game, depth-1, False)
                    best = max(best, score)
            return best
        # expected value
        else:
            empty = [(r, c) for r in range(4) for c in range(4) if game.board[r][c] == 0]
            if not empty:
                return self.evaluation_function(game.board)
            expected_value = 0
            for r, c in empty:
                for val, prob in [(2, 0.9), (4, 0.1)]:
                    new_game = copy.deepcopy(game)
                    new_game.board[r][c] = val
                    expected_value += prob * self.expectimax(new_game, depth-1, True) / len(empty)
            return expected_value

    # find best move to make based on expected values
    def get_best_move(self, game, depth=3):
        best_move = None
        best_value = float('-inf')

        for move in ['up', 'down', 'left', 'right']:
            new_game = copy.deepcopy(game)
            new_game.move(move)
            if new_game.board != game.board:
                value = self.expectimax(new_game, depth - 1, False)
                if value > best_value:
                    best_value = value
                    best_move = move

        return best_move


class Runner:
    def __init__(self, weight_sets, runs=100, depth=3):
        self.weight_sets = weight_sets
        self.runs = runs
        self.depth = depth

    def run_simulation(self, weights):
        max_tiles = []
        for i in range(self.runs):
            game = Game2048()
            expectimax = Expectimax(weights)
            while not game.game_over:
                move = expectimax.get_best_move(game, self.depth)
                if move:
                    game.move(move)
                else:
                    break
            max_tiles.append(max(max(row) for row in game.board))
        return max_tiles

    def run_analysis(self, weights):
        max_tiles = []
        for i in range(self.runs):
            game = Game2048()
            agent = Expectimax(weights)
            while not game.game_over:
                move = agent.get_best_move(game, self.depth)
                if move:
                    game.move(move)
                else:
                    break
            max_tiles.append(max(max(row) for row in game.board))
        tile_distribution = Counter(max_tiles)
        return tile_distribution

    def plot_results(self, distribution, weights):
        tiles = list(distribution.keys())
        frequencies = list(distribution.values())

        plt.bar(tiles, frequencies, width=50, color='skyblue', label='Tile Frequencies')
        plt.xlabel("Max Tile Achieved")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Max Tiles Achieved\nWeights: {weights}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def score_tiles(self, counter):
        score = 0
        for tile, count in counter.items():
            if tile < 32:
                continue
            points = (tile // 32) - 1
            score += points * count
        return score

    def graph_scores(self, scores):
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings, sorted_scores_values = zip(*sorted_scores)
        weight_sets = [str(rank) for rank in rankings]

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(sorted_scores_values))[::-1], sorted_scores_values[::-1],
                 tick_label=weight_sets[::-1], color='skyblue')
        plt.xlabel("Score")
        plt.ylabel("Weight Set")
        plt.title("Ranking of Weight Sets based on Scores")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def run(self):
        scores = {}
        for weights in self.weight_sets:
            print(f"Running analysis for weights: {weights}")
            distribution = self.run_analysis(weights)
            print(distribution)
            scores[tuple(weights)] = self.score_tiles(distribution)

            self.plot_results(distribution, weights)

        self.graph_scores(scores)

if __name__ == "__main__":
    weight_sets = [
        # Baseline
        (0, 0, 0, 0),

        # Varying corner weight
        (35, 0, 0, 0),
        (100, 0, 0, 0),
        (250, 0, 0, 0),
        (1200, 0, 0, 0),

        # Varying merging weight
        (0, 30, 0, 0),
        (0, 100, 0, 0),
        (0, 175, 0, 0),

        # Varying empty tiles weight
        (0, 0, 50, 0),
        (0, 0, 80, 0),
        (0, 0, 120, 0),

        # Varying full columns/rows weight
        (0, 0, 0, 100),
        (0, 0, 0, 200),
        (0, 0, 0, 400),

        # Varying weights around the best manual set of weights we found
        (0, 100, 80, 200),
        (30, 100, 80, 200),
        (180, 100, 80, 200),
        (1000, 100, 80, 200),

        (100, 0, 80, 200),
        (100, 30, 80, 200),
        (100, 180, 80, 200),
        (100, 1000, 80, 200),

        (100, 100, 0, 200),
        (100, 100, 20, 200),
        (100, 100, 160, 200),
        (100, 100, 1000, 200),

        (100, 100, 80, 0),
        (100, 100, 80, 100),
        (100, 100, 80, 300),
        (100, 100, 80, 1000),

        # Best manual set of weights we found
        (100, 100, 80, 150),
    ]

    runner = Runner(weight_sets, runs=25, depth=3)
    runner.run()