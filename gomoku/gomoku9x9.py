"""
Gomoku (五子棋) on a 9x9 Grid — Game Environment
==================================================
Rules: Two players (1 and -1) alternate placing pieces on a 9x9 board.
Win condition: 5 in a row horizontally, vertically, or diagonally.
"""

import numpy as np


class Gomoku9x9:
    """
    9x9 Gomoku game environment.

    Board encoding:
        0  = empty
        1  = player 1 (X)
       -1  = player 2 (O)

    State encoding for the neural network:
        shape (2, 9, 9)
        channel 0: current player's pieces
        channel 1: opponent's pieces
    """

    BOARD_SIZE = 9
    WIN_LENGTH = 5

    def __init__(self):
        self.board = None
        self.current_player = None
        self.done = None
        self.winner = None
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the board and return the initial state."""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self._get_state()

    def step(self, action):
        """
        Play action (int 0-80, row-major) for the current player.

        Returns:
            state  : np.array shape (2, 9, 9)
            reward : float — shaped reward
            done   : bool
            info   : dict
        """
        if self.done:
            raise RuntimeError("Game is already over. Call reset().")

        row, col = divmod(action, self.BOARD_SIZE)

        if self.board[row, col] != 0:
            # Illegal move — penalise heavily
            return self._get_state(), -10.0, True, {"illegal": True}

        self.board[row, col] = self.current_player

        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif len(self.get_valid_moves()) == 0:
            self.done = True
            self.winner = 0  # draw
            reward = 0.0
        else:
            # Intermediate reward shaping: reward building threats
            reward = self._threat_reward(row, col, self.current_player)

        info = {"winner": self.winner}
        self._switch_player()
        return self._get_state(), reward, self.done, info

    def get_valid_moves(self):
        """Return list of valid action indices (flattened)."""
        return [i for i in range(self.BOARD_SIZE ** 2)
                if self.board[i // self.BOARD_SIZE, i % self.BOARD_SIZE] == 0]

    def get_valid_mask(self):
        """Return boolean mask of shape (81,) — True where move is valid."""
        mask = np.zeros(self.BOARD_SIZE ** 2, dtype=bool)
        for m in self.get_valid_moves():
            mask[m] = True
        return mask

    def is_terminal(self):
        return self.done

    def get_board(self):
        """Return raw board (9×9 numpy array)."""
        return self.board.copy()

    def get_current_player(self):
        return self.current_player

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _get_state(self):
        """
        Encode board as (2, 9, 9) tensor from the perspective of
        the *current* player.
        """
        state = np.zeros((2, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        return state

    def get_state_for_player(self, player):
        """Encode board from a specific player's perspective."""
        state = np.zeros((2, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        state[0] = (self.board == player).astype(np.float32)
        state[1] = (self.board == -player).astype(np.float32)
        return state

    # ------------------------------------------------------------------
    # Win detection
    # ------------------------------------------------------------------

    def _check_win(self, player):
        b = self.board
        n = self.BOARD_SIZE
        w = self.WIN_LENGTH

        # Horizontal
        for r in range(n):
            for c in range(n - w + 1):
                if all(b[r, c + k] == player for k in range(w)):
                    return True

        # Vertical
        for r in range(n - w + 1):
            for c in range(n):
                if all(b[r + k, c] == player for k in range(w)):
                    return True

        # Diagonal ↘
        for r in range(n - w + 1):
            for c in range(n - w + 1):
                if all(b[r + k, c + k] == player for k in range(w)):
                    return True

        # Diagonal ↙
        for r in range(n - w + 1):
            for c in range(w - 1, n):
                if all(b[r + k, c - k] == player for k in range(w)):
                    return True

        return False

    def _count_in_direction(self, row, col, dr, dc, player):
        """Count consecutive pieces in one direction from (row, col)."""
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def _threat_reward(self, row, col, player):
        """
        Small shaping reward for placing a piece that creates a long run.
        +0.1  for creating a 4-in-a-row (one step from winning)
        +0.05 for creating a 3-in-a-row
        Applied only if game is not over yet.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            run = (1
                   + self._count_in_direction(row, col, dr, dc, player)
                   + self._count_in_direction(row, col, -dr, -dc, player))
            if run >= 4:
                return 0.1
            if run >= 3:
                return 0.05
        return 0.0

    def _switch_player(self):
        self.current_player *= -1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print(f"\n  Current player: {'X' if self.current_player == 1 else 'O'}")
        print("   " + " ".join(f"{c:2}" for c in range(self.BOARD_SIZE)))
        for r in range(self.BOARD_SIZE):
            row_str = "  ".join(symbols[int(self.board[r, c])]
                                for c in range(self.BOARD_SIZE))
            print(f"{r:2} {row_str}")
        print()

    def clone(self):
        """Return a deep copy of this environment."""
        new_env = Gomoku9x9()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        return new_env


# ------------------------------------------------------------------
# Random Agent
# ------------------------------------------------------------------

class RandomAgent:
    """Plays uniformly at random from valid moves."""

    def predict(self, state, env):
        valid = env.get_valid_moves()
        action = np.random.choice(valid)
        row, col = divmod(action, Gomoku9x9.BOARD_SIZE)
        return row, col

    def select_action(self, env):
        valid = env.get_valid_moves()
        return np.random.choice(valid)


# ------------------------------------------------------------------
# Smart Agent (1-step lookahead)
# ------------------------------------------------------------------

class SmartAgent:
    """
    Looks 1 step ahead:
      1. Take an immediate win if available.
      2. Block opponent's immediate win.
      3. Otherwise, prefer center/near-center cells, else random.
    """

    def select_action(self, env):
        valid = env.get_valid_moves()
        player = env.current_player
        board = env.board
        n = Gomoku9x9.BOARD_SIZE

        # 1. Can I win immediately?
        for action in valid:
            r, c = divmod(action, n)
            board[r, c] = player
            won = env._check_win(player)
            board[r, c] = 0
            if won:
                return action

        # 2. Must I block immediately?
        for action in valid:
            r, c = divmod(action, n)
            board[r, c] = -player
            opp_wins = env._check_win(-player)
            board[r, c] = 0
            if opp_wins:
                return action

        # 3. Prefer cells closer to the center
        center = n // 2
        valid_sorted = sorted(
            valid,
            key=lambda a: abs(a // n - center) + abs(a % n - center)
        )
        # Add some randomness among top candidates
        top = valid_sorted[:max(1, len(valid_sorted) // 3)]
        return np.random.choice(top)
