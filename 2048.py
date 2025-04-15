import random
import copy

#from mpmath.matrices.matrices import rowsep
import matplotlib.pyplot as plt

class Game2048:
    def __init__(self):
        self.board = [[0]*4 for _ in range(4)]
        self.score = 0
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
                self.score += row[i]
                row[i+1] = 0
        return row

    def final_merge(self, row):
        return self.clean_up(self.merge(self.clean_up(row)))

    def move_left(self):
        self.board = [self.final_merge(row) for row in self.board]

    def move_right(self):
        self.board = [self.final_merge(row[::-1])[::-1] for row in self.board]

    def move_up(self):
        transposed = list(map(list, zip(*self.board)))
        merged = [self.final_merge(row) for row in transposed]
        self.board = [list(row) for row in zip(*merged)]

    def move_down(self):
        transposed = list(map(list, zip(*self.board)))
        merged = [self.final_merge(row[::-1])[::-1] for row in transposed]
        self.board = [list(row) for row in zip(*merged)]

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
        else:
            return
        if self.board != old_board:
            self.spawn_number_tile()
        elif not self.can_move():
            self.game_over = True

    def can_move(self):
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:
                    return True
                if col < 3 and self.board[row][col] == self.board[row][col+1]:
                    return True
                if row < 3 and self.board[row][col] == self.board[row+1][col]:
                    return True
        return False


class Expectimax:
    def __init__(self, weights):
        self.weights = weights  # (corner, merge, empty)

    def is_max_tile_in_corner(self, board):
        max_tile = max(max(row) for row in board)
        return max_tile in [board[0][0], board[0][3], board[3][0], board[3][3]]

    def count_potential_merges(self, board):
        merges = 0
        for row in board:
            row = [x for x in row if x != 0]
            for i in range(len(row) - 1):
                if row[i] == row[i+1]:
                    merges += 1
        for col in zip(*board):
            col = [x for x in col if x != 0]
            for i in range(len(col) - 1):
                if col[i] == col[i+1]:
                    merges += 1
        return merges

    def count_empty_tiles(self, board):
        return sum(cell == 0 for row in board for cell in row)

    def evaluation_function(self, board):
        w_corner, w_merge, w_empty = self.weights
        corner_score = w_corner if self.is_max_tile_in_corner(board) else 0
        merge_score = self.count_potential_merges(board) * w_merge
        empty_score = self.count_empty_tiles(board) * w_empty
        return corner_score + merge_score + empty_score

    def expectimax(self, game, depth, is_max_turn):
        if depth == 0 or game.game_over:
            return self.evaluation_function(game.board)
        if is_max_turn:
            best = float('-inf')
            for move in ['up', 'down', 'left', 'right']:
                new_game = copy.deepcopy(game)
                new_game.move(move)
                if new_game.board != game.board:
                    score = self.expectimax(new_game, depth-1, False)
                    best = max(best, score)
            return best
        else:
            empty = [(r, c) for r in range(4) for c in range(4) if game.board[r][c] == 0]
            if not empty:
                return self.evaluation_function(game.board)
            expect = 0
            for r, c in empty:
                for val, prob in [(2, 0.9), (4, 0.1)]:
                    new_game = copy.deepcopy(game)
                    new_game.board[r][c] = val
                    expect += prob * self.expectimax(new_game, depth-1, True) / len(empty)
            return expect

    def get_best_move(self, game, depth=3):
        best_move = None
        best_score = float('-inf')
        for move in ['up', 'down', 'left', 'right']:
            new_game = copy.deepcopy(game)
            new_game.move(move)
            if new_game.board != game.board:
                score = self.expectimax(new_game, depth-1, False)
                if score > best_score:
                    best_score = score
                    best_move = move
        return best_move


def run_simulation(weights, runs=10, depth=3):
    scores = []
    max_tiles = []
    for _ in range(runs):
        game = Game2048()
        ai = Expectimax(weights)
        while not game.game_over:
            move = ai.get_best_move(game, depth)
            if move:
                game.move(move)
            else:
                break
        scores.append(game.score)
        max_tiles.append(max(max(row) for row in game.board))
    return scores, max_tiles


if __name__ == "__main__":
    weight_sets = [
        (1500, 50, 100),
        (1000, 30, 120),
        (2000, 80, 80),
        (0, 100, 150)
    ]

    print("Running simulations...")
    for weights in weight_sets:
        print(f"Simulating weights: {weights}")
        scores, _ = run_simulation(weights, runs=5, depth=2)  
        plt.plot(scores, label=f"Weights: {weights}")

    print("Done. Plotting...")
    plt.title("2048 AI Scores by Weight Combination")
    plt.xlabel("Simulation #")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
