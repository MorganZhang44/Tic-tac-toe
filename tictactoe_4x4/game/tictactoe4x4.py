"""
Connect-4 on a 4x4 Grid — Game Environment
=============================================
Rules: Two players (1 and -1) alternate placing pieces on a 4x4 board.
Win condition: 4 in a row horizontally, vertically, or diagonally.
"""

import numpy as np


class TicTacToe4x4:
    """
    4x4 Connect-4 game environment.

    Board encoding:
        0  = empty
        1  = player 1
       -1  = player 2 (opponent)

    State encoding for the neural network:
        shape (2, 4, 4)
        channel 0: current player's pieces
        channel 1: opponent's pieces
    """

    BOARD_SIZE = 4
    WIN_LENGTH = 4  # need 4 in a row to win

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
        self.current_player = 1  # player 1 goes first
        self.done = False
        self.winner = None
        return self._get_state()

    def step(self, action):
        """
        Play action (int 0-15, row-major) for the current player.

        Returns:
            state  : np.array shape (2,4,4)
            reward : float  (+1 win, -1 loss, 0 otherwise)
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
            reward = 0.0  # Removing the 'safe' draw reward to force aggressive play
        else:
            reward = 0.0

        info = {"winner": self.winner}
        self._switch_player()
        return self._get_state(), reward, self.done, info

    def get_valid_moves(self):
        """Return list of valid action indices (flattened)."""
        return [i for i in range(self.BOARD_SIZE ** 2)
                if self.board[i // self.BOARD_SIZE, i % self.BOARD_SIZE] == 0]

    def get_valid_mask(self):
        """Return boolean mask of shape (16,) — True where move is valid."""
        mask = np.zeros(self.BOARD_SIZE ** 2, dtype=bool)
        for m in self.get_valid_moves():
            mask[m] = True
        return mask

    def is_terminal(self):
        return self.done

    def get_board(self):
        """Return raw board (4×4 numpy array)."""
        return self.board.copy()

    def get_current_player(self):
        return self.current_player

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _get_state(self):
        """
        Encode board as (2, 4, 4) tensor from the perspective of
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
    # Win / draw detection
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

    def _switch_player(self):
        self.current_player *= -1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        print(f"\n  Current player: {'X' if self.current_player == 1 else 'O'}")
        print("  " + " ".join(str(c) for c in range(self.BOARD_SIZE)))
        for r in range(self.BOARD_SIZE):
            row_str = "  ".join(symbols[int(self.board[r, c])]
                                for c in range(self.BOARD_SIZE))
            print(f"{r} {row_str}")
        print()

    def clone(self):
        """Return a deep copy of this environment."""
        new_env = TicTacToe4x4()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        return new_env


# ------------------------------------------------------------------
# Convenience: random agent (for baseline comparisons)
# ------------------------------------------------------------------

class RandomAgent:
    """Plays uniformly at random from valid moves."""

    def predict(self, state, env):
        valid = env.get_valid_moves()
        action = np.random.choice(valid)
        row, col = divmod(action, TicTacToe4x4.BOARD_SIZE)
        return row, col

    def select_action(self, env):
        valid = env.get_valid_moves()
        return np.random.choice(valid)

# ------------------------------------------------------------------
# Heuristic: Smart agent (for aggressive training baseline)
# ------------------------------------------------------------------

class SmartAgent:
    """Looks 1 step ahead: takes immediate wins, blocks immediate losses, else random."""

    def select_action(self, env):
        valid = env.get_valid_moves()
        player = env.current_player
        
        # 1. Can I win immediately?
        for action in valid:
            row, col = divmod(action, TicTacToe4x4.BOARD_SIZE)
            env.board[row, col] = player
            if env._check_win(player):
                env.board[row, col] = 0
                return action
            env.board[row, col] = 0
                
        # 2. Must I block immediately?
        for action in valid:
            row, col = divmod(action, TicTacToe4x4.BOARD_SIZE)
            env.board[row, col] = -player
            if env._check_win(-player):
                env.board[row, col] = 0
                return action
            env.board[row, col] = 0
                
        # 3. Otherwise play randomly (could add center preference but random is fine for exploration)
        return np.random.choice(valid)
