import random
import copy

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
            print("".join(f"{num or '.':>5}" for num in row))
        print("\n")

if __name__ == "__main__":
    game = Game2048()
    game.print_board()

    while not game.game_over:
        move = input("Move (w/a/s/d): ").lower()
        dir_map = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right'}

        if move in dir_map:
            game.move(dir_map[move])
            game.print_board()
        else:
            print("Invalid move")

    print("Final score:", game.score)