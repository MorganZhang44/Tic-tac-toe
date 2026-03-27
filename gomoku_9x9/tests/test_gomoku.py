"""
Unit Tests — 9x9 Gomoku Game Environment
==========================================
Run with: python -m pytest tests/test_gomoku.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.gomoku9x9 import Gomoku9x9, RandomAgent


class TestGomokuBoard:
    def setup_method(self):
        self.env = Gomoku9x9()

    def test_reset_returns_clean_board(self):
        state = self.env.reset()
        assert state.shape == (2, 9, 9)
        assert state.sum() == 0

    def test_all_moves_valid_at_start(self):
        self.env.reset()
        valid = self.env.get_valid_moves()
        assert len(valid) == 81

    def test_step_reduces_valid_moves(self):
        self.env.reset()
        self.env.step(0)
        valid = self.env.get_valid_moves()
        assert len(valid) == 80
        assert 0 not in valid

    def test_illegal_move_ends_game_with_penalty(self):
        self.env.reset()
        self.env.step(0)          # player 1
        self.env.step(1)          # player 2
        self.env.current_player = 1
        _, reward, done, _ = self.env.step(0)
        assert reward == -10.0
        assert done

    def test_player_switch(self):
        self.env.reset()
        assert self.env.current_player == 1
        self.env.step(0)
        assert self.env.current_player == -1

    def test_horizontal_win(self):
        env = self.env
        env.reset()
        env.board[0, :5] = 1   # 5 in a row horizontally
        assert env._check_win(1)
        assert not env._check_win(-1)

    def test_vertical_win(self):
        env = self.env
        env.reset()
        env.board[:5, 0] = 1
        assert env._check_win(1)

    def test_diagonal_win(self):
        env = self.env
        env.reset()
        for i in range(5):
            env.board[i, i] = 1
        assert env._check_win(1)

    def test_anti_diagonal_win(self):
        env = self.env
        env.reset()
        for i in range(5):
            env.board[i, 8 - i] = 1
        assert env._check_win(1)

    def test_no_win_with_four(self):
        """4 in a row should NOT win in Gomoku."""
        env = self.env
        env.reset()
        env.board[0, :4] = 1
        assert not env._check_win(1)

    def test_state_encoding_shape(self):
        env = self.env
        env.reset()
        state = env._get_state()
        assert state.shape == (2, 9, 9)
        assert state.dtype == np.float32

    def test_state_perspective(self):
        env = self.env
        env.reset()
        env.step(0)   # player 1 moves to (0,0)
        # After move, current_player is -1
        state = env._get_state()
        # Channel 1 (opponent) should show player 1's piece at (0,0)
        assert state[1, 0, 0] == 1.0

    def test_valid_mask_length(self):
        env = self.env
        env.reset()
        mask = env.get_valid_mask()
        assert mask.shape == (81,)
        assert mask.sum() == 81

    def test_clone_independence(self):
        env = self.env
        env.reset()
        env.step(5)
        clone = env.clone()
        assert np.array_equal(env.board, clone.board)
        assert env.current_player == clone.current_player
        clone.step(0)
        assert not np.array_equal(env.board, clone.board)

    def test_full_game_terminates(self):
        """Random self-play must eventually terminate."""
        env = self.env
        env.reset()
        agent = RandomAgent()
        done  = False
        steps = 0
        while not done:
            action = agent.select_action(env)
            _, _, done, _ = env.step(action)
            steps += 1
            assert steps <= 82   # at most 81 cells + 1 guard

    def test_win_detection_via_step(self):
        """Stepping into a 5-in-a-row returns reward=1 and done=True."""
        env = self.env
        env.reset()
        # Manually put 4 pieces for player 1
        for col in range(4):
            env.board[0, col] = 1
        env.current_player = 1
        _, reward, done, info = env.step(4)   # complete 5 in row 0
        assert done
        assert reward == 1.0
        assert info["winner"] == 1


class TestRandomAgent:
    def test_always_returns_valid_action(self):
        env   = Gomoku9x9()
        agent = RandomAgent()
        for _ in range(50):
            env.reset()
            valid  = env.get_valid_moves()
            action = agent.select_action(env)
            assert action in valid

    def test_predict_returns_valid_tuple(self):
        env   = Gomoku9x9()
        agent = RandomAgent()
        env.reset()
        state = env._get_state()
        move  = agent.predict(state, env)
        assert isinstance(move, tuple) and len(move) == 2
        r, c = move
        assert 0 <= r < 9 and 0 <= c < 9
