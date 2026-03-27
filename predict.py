"""
predict.py — Coursework-Required Prediction API
=================================================
This file exposes the clean interface required by the coursework spec:

    predict(board_state) -> (x, y)

where:
    board_state : np.ndarray of shape (4, 4)
                  values: 0 = empty, 1 = current player, -1 = opponent
    (x, y)      : (row, col) of the chosen move

Usage:
    # Import and use
    from predict import predict
    move = predict(board_state)

    # Self-test
    python predict.py --test
"""

import os
import numpy as np
import torch

from agent.dqn_agent import DQNAgent

# ── Global model (loaded once) ────────────────────────────────────────
_agent: DQNAgent = None
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__),
                              "weights", "model_weights.pth")


def _load_agent():
    global _agent
    if _agent is None:
        _agent = DQNAgent()
        if os.path.exists(_WEIGHTS_PATH):
            _agent.load(_WEIGHTS_PATH)
            _agent.policy_net.eval()
        else:
            print(f"[WARNING] Weights not found at '{_WEIGHTS_PATH}'. "
                  "Using untrained agent — run train.py first!")
    return _agent


def predict(board_state: np.ndarray):
    """
    Predict the best move for the current player.

    Args:
        board_state : np.ndarray shape (4, 4)
                      0 = empty, 1 = your pieces, -1 = opponent's pieces

    Returns:
        (row, col) : tuple[int, int]  — best move coordinates
    """
    agent = _load_agent()
    board = np.asarray(board_state, dtype=np.float32)
    return agent.predict(board)


# ── CLI self-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run a quick self-test of the predict() function")
    args = parser.parse_args()

    if args.test:
        print("\n── predict.py self-test ──")

        # Test 1: empty board
        board = np.zeros((4, 4), dtype=np.float32)
        move = predict(board)
        assert isinstance(move, tuple) and len(move) == 2, \
            f"Expected (row, col) tuple, got {move}"
        r, c = move
        assert 0 <= r < 4 and 0 <= c < 4, \
            f"Move {move} is out of board bounds"
        assert board[r, c] == 0, \
            f"Move {move} targets a non-empty cell!"
        print(f"  [PASS] Empty board → move={move}")

        # Test 2: nearly full board
        board2 = np.array([
            [ 1, -1,  1, -1],
            [-1,  1, -1,  1],
            [ 1, -1,  1,  0],   # only (2,3) is free
            [-1,  1, -1,  1],
        ], dtype=np.float32)
        move2 = predict(board2)
        assert move2 == (2, 3), \
            f"Expected (2, 3) as sole valid move, got {move2}"
        print(f"  [PASS] Nearly full board → move={move2}")

        # Test 3: blocking a winning threat
        board3 = np.array([
            [ 1,  1,  1,  0],   # player 1 about to win at (0,3)
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
        ], dtype=np.float32)
        move3 = predict(board3)
        r3, c3 = move3
        assert board3[r3, c3] == 0, f"Move {move3} targets non-empty cell!"
        print(f"  [PASS] Threat board → move={move3}")

        print("\n  ✓ All tests passed!\n")
    else:
        # Interactive demo
        print("\nRunning interactive demo…")
        board = np.zeros((4, 4), dtype=np.float32)
        move = predict(board)
        print(f"  predict(empty_board) = {move}")
