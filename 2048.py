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
        empty_tiles = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if not empty_tiles:
            return
        r, c = random.choice(empty_tiles)
        self.board[r][c] = random.choices([2, 4], weights=[0.9, 0.1])[0]

    def compress(self, row):
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

    def move_row_left(self, row):
        row = self.compress(row)
        row = self.merge(row)
        row = self.compress(row)
        return row

    def move_left(self):
        self.board = [self.move_row_left(row) for row in self.board]

    def move_right(self):
        new_board = []
        for row in self.board:
            reversed_row = row[::-1]
            moved_row = self.move_row_left(reversed_row)
            new_board.append(moved_row[::-1])
        self.board = new_board

    def move_up(self):
        transposed = [[self.board[r][c] for r in range(4)] for c in range(4)]
        moved = [self.move_row_left(row) for row in transposed]
        self.board = [[moved[c][r] for c in range(4)] for r in range(4)]

    def move_down(self):
        transposed = [[self.board[r][c] for r in range(4)][::-1] for c in range(4)]
        moved = [self.move_row_left(row) for row in transposed]
        unflipped = [row[::-1] for row in moved]
        self.board = [[unflipped[c][r] for c in range(4)] for r in range(4)]

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
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    return True
                if c+1 < 4 and self.board[r][c] == self.board[r][c+1]:
                    return True
                if r+1 < 4 and self.board[r][c] == self.board[r+1][c]:
                    return True
        return False

    def print_board(self):
        print("\nScore:", self.score)
        print("-" * (4 * 6))
        for row in self.board:
            print("".join(f"{num or '.':>5}" for num in row))
        print("-" * (4 * 6))

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