"""
predict_gomoku.py — Coursework-Required Prediction API (Gomoku)
================================================================
Exposes the clean interface required by the coursework spec:

    predict(board_state) -> (x, y)

where:
    board_state : np.ndarray of shape (9, 9)
                  values: 0 = empty, 1 = current player, -1 = opponent
    (x, y)      : (row, col) of the chosen move

Usage:
    # Import and use
    from predict_gomoku import predict
    # Device is automatically selected (CPU fallback if GPU lacks kernels)
    move = predict(board_state)

    # Self-test
    python predict_gomoku.py --test
"""

import os
import numpy as np

from models.cnn_standard.agent import GomokuDQNAgent

# ── Global model (loaded once) ────────────────────────────────────────
_agent: GomokuDQNAgent = None
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__),
                              "models", "cnn_standard", "weights.pth")


def _load_agent():
    global _agent
    if _agent is None:
        _agent = GomokuDQNAgent()
        if os.path.exists(_WEIGHTS_PATH):
            _agent.load(_WEIGHTS_PATH)
            _agent.policy_net.eval()
        else:
            print(f"[WARNING] Weights not found at '{_WEIGHTS_PATH}'. "
                  "Using untrained agent — run train_gomoku.py first!")
    return _agent


def predict(board_state: np.ndarray):
    """
    Predict the best move for the current player on a 9x9 Gomoku board.

    Args:
        board_state : np.ndarray shape (9, 9)
                      0 = empty, 1 = your pieces, -1 = opponent's pieces

    Returns:
        (row, col) : tuple[int, int] — best move coordinates
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
        print("\n── predict_gomoku.py self-test ──")

        # Test 1: empty board
        board = np.zeros((9, 9), dtype=np.float32)
        move  = predict(board)
        assert isinstance(move, tuple) and len(move) == 2, \
            f"Expected (row, col) tuple, got {move}"
        r, c = move
        assert 0 <= r < 9 and 0 <= c < 9, f"Move {move} is out of bounds"
        assert board[r, c] == 0, f"Move {move} targets a non-empty cell!"
        print(f"  [PASS] Empty board → move={move}")

        # Test 2: nearly full board (one cell left)
        board2 = np.ones((9, 9), dtype=np.float32)
        board2 *= -1
        board2[1, 1] = 0   # only (1,1) is empty
        board2[0, :] = 0   # clear row 0 (avoid accidental 5 in a row)
        board2[0, 0] = 0
        board2[4, 4] = 0   # leave 2 cells
        board2[1, 1] = 0
        # Simpler: just one empty cell
        b = np.zeros((9, 9), dtype=np.float32)
        for i in range(9):
            for j in range(9):
                if not (i == 4 and j == 4):
                    b[i, j] = 1 if (i + j) % 2 == 0 else -1
        move2 = predict(b)
        assert move2 == (4, 4), f"Expected (4, 4) as sole valid move, got {move2}"
        print(f"  [PASS] One-cell board → move={move2}")

        # Test 3: about-to-win scenario (4 in a row)
        board3 = np.zeros((9, 9), dtype=np.float32)
        board3[0, 0] = board3[0, 1] = board3[0, 2] = board3[0, 3] = 1  # 4 in row 0
        move3 = predict(board3)
        r3, c3 = move3
        assert board3[r3, c3] == 0, f"Move {move3} is on an occupied cell!"
        print(f"  [PASS] Near-win board → move={move3}")

        print("\n  ✓ All tests passed!\n")
    else:
        print("\nRunning interactive demo…")
        board = np.zeros((9, 9), dtype=np.float32)
        move  = predict(board)
        print(f"  predict(empty 9x9 board) = {move}")
