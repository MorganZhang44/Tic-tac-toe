import numpy as np
import sys
import os

# Create mock environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gomoku.dqn_agent_gomoku import GomokuDQNAgent

agent = GomokuDQNAgent(device="cpu")

# build exactly the board from the user's screenshot
# user placed pieces:
X_moves = [(2, 4), (3, 3), (3, 4), (3, 6), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
O_moves = [(0, 6), (1, 4), (1, 7), (2, 3), (2, 5), (3, 2), (4, 2), (5, 0), (6, 3), (6, 4), (8, 8)]

# Let's say it's O's turn (AI). Human just played 5,4. Human has 5,1 to 5,4. 
# 5,5 is empty. AI SHOULD play 5,5!
board = np.zeros((9, 9), dtype=np.float32)
for r, c in X_moves: board[r, c] = 1    # Human
for r, c in O_moves: board[r, c] = -1   # AI

# play_gomoku.py logic for AI's turn (player = -1):
player = -1
board_for_predict = board * player 
# Now 1 corresponds to O (my pieces), -1 to X (opp pieces)

print("Testing predict...")
print("board_for_predict:")
print(board_for_predict)

# Inside predict(board_state):
state = np.zeros((2, 9, 9), dtype=np.float32)
state[0] = (board_for_predict == 1).astype(np.float32)
state[1] = (board_for_predict == -1).astype(np.float32)
valid_mask = (board_for_predict.flatten() == 0)

print("state[0] (AI pieces O):")
print(state[0].astype(int))
print("state[1] (Human pieces X):")
print(state[1].astype(int))

# select_action with greedy=True
my_pieces = state[0]
opp_pieces = state[1]

print("\n--- Running step 2 block logic ---")
opp_wins = agent._find_winning_actions(opp_pieces, my_pieces)
print("opp_wins returned:", opp_wins)

# If it failed, let's step into _find_winning_actions manually
n = 9
wins = []
mask = ((opp_pieces + my_pieces) == 0).flatten()
for action in np.where(mask)[0]:
    r, c = divmod(action, n)
    # simulate action 5,5
    if r == 5 and c == 5:
        print("Checking 5,5...")
        opp_pieces[r, c] = 1
        print("opp_pieces row 5 slice:", opp_pieces[5, 1:6])
        val = opp_pieces[5, 1:6].sum()
        print("sum:", val)
        print("val == 5:", val == 5)
        print("type(val):", type(val))
        print("_check_win(opp_pieces) inside loop:", agent._check_win(opp_pieces))
        
        # Test trace inside check_win
        print("Let's do manual check win row 5:")
        for r_check in range(n):
            for c_check in range(n - 4):
                s = opp_pieces[r_check, c_check:c_check+5].sum()
                if s == 5:
                    print(f"FOUND 5 IN A ROW at r={r_check}, c_start={c_check}")
                    
        opp_pieces[r, c] = 0

print("\nFinal action chosen by predict:", agent.predict(board_for_predict))
