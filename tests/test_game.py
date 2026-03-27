"""
Unit Tests — 4x4 Connect-4 Game Environment
=============================================
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game.tictactoe4x4 import TicTacToe4x4, RandomAgent


class TestBoard:
    def setup_method(self):
        self.env = TicTacToe4x4()

    def test_reset_returns_clean_board(self):
        state = self.env.reset()
        assert state.shape == (2, 4, 4)
        assert state.sum() == 0

    def test_all_moves_valid_at_start(self):
        self.env.reset()
        valid = self.env.get_valid_moves()
        assert len(valid) == 16

    def test_step_reduces_valid_moves(self):
        self.env.reset()
        self.env.step(0)
        valid = self.env.get_valid_moves()
        assert len(valid) == 15
        assert 0 not in valid

    def test_illegal_move_ends_game_with_penalty(self):
        self.env.reset()
        self.env.step(0)           # player 1
        self.env.step(1)           # player 2
        # Try to play on cell 0, which is already taken
        # We have to manually set current player back to test harshly
        self.env.current_player = 1
        _, reward, done, info = self.env.step(0)
        assert reward == -10.0
        assert done

    def test_player_switch(self):
        self.env.reset()
        assert self.env.current_player == 1
        self.env.step(0)  # move for player 1
        assert self.env.current_player == -1

    def test_horizontal_win(self):
        """Row 0, cols 0-3 for player 1 → win."""
        env = self.env
        env.reset()
        # Lay out: X X X X on row 0 (player 1 goes, player 2 goes elsewhere)
        moves = [(0, 1), (4, -1), (1, 1), (5, -1), (2, 1), (6, -1), (3, 1)]
        actions_players = [(0, 1), (4, -1), (1, 1), (5, -1), (2, 1), (6, -1), (3, 1)]
        # Simpler: just set the board manually and check _check_win
        env.board[0, :] = 1
        assert env._check_win(1)
        assert not env._check_win(-1)

    def test_vertical_win(self):
        env = self.env
        env.reset()
        env.board[:, 0] = 1
        assert env._check_win(1)

    def test_diagonal_win(self):
        env = self.env
        env.reset()
        for i in range(4):
            env.board[i, i] = 1
        assert env._check_win(1)

    def test_anti_diagonal_win(self):
        env = self.env
        env.reset()
        for i in range(4):
            env.board[i, 3 - i] = 1
        assert env._check_win(1)

    def test_no_win_partial(self):
        env = self.env
        env.reset()
        env.board[0, 0] = 1
        env.board[0, 1] = 1
        env.board[0, 2] = 1
        assert not env._check_win(1)  # only 3 in a row

    def test_draw_detection(self):
        env = self.env
        env.reset()
        # Fill the board with no 4-in-a-row for either player.
        # This pattern breaks all diagonal and row/column runs of 4.
        pattern = [
            [ 1,  1, -1, -1],
            [-1, -1,  1,  1],
            [ 1,  1, -1, -1],
            [-1, -1,  1,  1],
        ]
        env.board = np.array(pattern, dtype=np.float32)
        assert len(env.get_valid_moves()) == 0
        assert not env._check_win(1)
        assert not env._check_win(-1)

    def test_state_encoding_shape(self):
        env = self.env
        env.reset()
        state = env._get_state()
        assert state.shape == (2, 4, 4)
        assert state.dtype == np.float32

    def test_state_perspective(self):
        env = self.env
        env.reset()
        env.step(0)  # player 1 plays corner
        # After move, current_player is -1
        state = env._get_state()
        # Channel 0 is -1's pieces (none yet), channel 1 is 1's piece
        assert state[1, 0, 0] == 1.0  # player 1 at (0,0) is opponent now

    def test_valid_mask_length(self):
        env = self.env
        env.reset()
        mask = env.get_valid_mask()
        assert mask.shape == (16,)
        assert mask.sum() == 16

    def test_clone(self):
        env = self.env
        env.reset()
        env.step(5)
        clone = env.clone()
        assert np.array_equal(env.board, clone.board)
        assert env.current_player == clone.current_player
        # Ensure independence
        clone.step(0)
        assert not np.array_equal(env.board, clone.board)

    def test_full_game_terminates(self):
        """Random self-play should always terminate."""
        env = self.env
        env.reset()
        agent = RandomAgent()
        done = False
        steps = 0
        while not done:
            action = agent.select_action(env)
            _, _, done, _ = env.step(action)
            steps += 1
            assert steps <= 16 + 1  # board has 16 cells max


class TestRandomAgent:
    def test_always_valid(self):
        env = TicTacToe4x4()
        env.reset()
        agent = RandomAgent()
        for _ in range(100):
            env.reset()
            valid = env.get_valid_moves()
            action = agent.select_action(env)
            assert action in valid

    def test_predict_returns_tuple(self):
        env = TicTacToe4x4()
        env.reset()
        agent = RandomAgent()
        state = env._get_state()
        move = agent.predict(state, env)
        assert isinstance(move, tuple) and len(move) == 2
        r, c = move
        assert 0 <= r < 4 and 0 <= c < 4
