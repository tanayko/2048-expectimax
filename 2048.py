import random
import copy

from mpmath.matrices.matrices import rowsep


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
        for i in range(4 - 1):
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
        else:
            print("Invalid move")
            return

        if self.board != old_board:
            self.spawn_number_tile()
        elif not self.can_move():
            self.game_over = True

    def can_move(self):
        for row in range(4):
            for column in range(4):
                if self.board[row][column] == 0:
                    return True
                if column + 1 < 4 and self.board[row][column] == self.board[row][column+1]:
                    return True
                if row + 1 < 4 and self.board[row][column] == self.board[row+1][column]:
                    return True
        return False

    def print_board(self):
        print("\nScore:", self.score)
        print("\n")
        for row in self.board:
            print("".join(f"{num or '-':>5}" for num in row))
        print("\n")


# EXPECTIMAX
class Expectimax:
    # helper functions
    def is_max_tile_in_corner(self, board):
        max_tile = max(max(row) for row in board)
        corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
        return max_tile in corners

    def count_potential_merges(self, board):
        potential_merges = 0
        for row in board:
            stripped = [num for num in row if num != 0]
            for i in range(len(stripped) - 1):
                if stripped[i] == stripped[i + 1]:
                    potential_merges += 1
        for c in range(4):
            col = [board[r][c] for r in range(4)]
            stripped = [num for num in col if num != 0]
            for i in range(len(stripped) - 1):
                if stripped[i] == stripped[i + 1]:
                    potential_merges += 1

        return potential_merges

    def count_empty_tiles(self, board):
        return sum(cell == 0 for row in board for cell in row)

    # run on terminal nodes
    def evaluation_function(self, board):
        corner_feature = 1500 if self.is_max_tile_in_corner(board) else 0
        potential_merge_feature = self.count_potential_merges(board) * 50
        empty_tile_feature = self.count_empty_tiles(board) * 100

        return corner_feature + potential_merge_feature + empty_tile_feature

    def expectimax(self, game, depth, is_max_turn):
        # terminal node
        if depth == 0 or game.game_over:
            return self.evaluation_function(game.board)

        # max over all possible moves
        if is_max_turn:
            maximum = float('-inf')
            for move in ['up', 'down', 'left', 'right']:
                new_game = copy.deepcopy(game)
                new_game.move(move)
                if new_game.board != game.board:
                    score = self.expectimax(new_game, depth - 1, False)
                    maximum = max(maximum, score)
            return maximum
        # expected value
        else:
            empty = [(row, column) for row in range(4) for column in range(4) if game.board[row][column] == 0]

            expected_value = 0
            for r, c in empty:
                for val, prob in [(2, 0.9), (4, 0.1)]:
                    new_game = copy.deepcopy(game)
                    new_game.board[r][c] = val
                    expected_value += prob * (self.expectimax(new_game, depth - 1, True) / len(empty))
            return expected_value

    # find best move to make based on expected values
    def get_best_move(self, game, depth=3):
        best_score = float('-inf')
        best_move = None
        for move in ['up', 'down', 'left', 'right']:
            new_game = copy.deepcopy(game)
            new_game.move(move)
            if new_game.board != game.board:
                score = self.expectimax(new_game, depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_move = move
        return best_move

if __name__ == "__main__":
    game = Game2048()
    game.print_board()

    expectimax = Expectimax()

    while not game.game_over:
        move = expectimax.get_best_move(game)
        if move:
            print(f"Best Move: {move}")
            game.move(move)
            game.print_board()
        else:
            print("Game Over")
            break

    print("Final score:", game.score)