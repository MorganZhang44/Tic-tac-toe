import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gomoku.dqn_agent_gomoku import GomokuDQNAgent

agent = GomokuDQNAgent(device="cpu")

my_pieces = np.zeros((9, 9), dtype=np.float32)
opp_pieces = np.zeros((9, 9), dtype=np.float32)

# Reconstruct board from Screenshot 3, IMMEDIATELY BEFORE Human plays (6, 4) to fork!
# X pieces (Human) BEFORE (6, 4) is played:
X_moves = [(3, 4), (4, 3), (4, 4), (4, 5), (4, 6), (5, 1), (5, 3), (5, 4), (6, 2), (6, 3), (6, 5)] 
# I assume X had 6,2 6,3 6,5 in the horizontal line, and 3,4 4,4 5,4 in the vertical.

# O pieces (AI) BEFORE (6, 4) is played:
O_moves = [(0, 6), (1, 4), (1, 7), (2, 3), (2, 5), (3, 2), (3, 7), (4, 2), (5, 2), (5, 5), (6, 1)] 
# Notice there is an O at (5, 2) blocking X's diagonal? Wait, I saw O at (5, 2)!

for r, c in X_moves: opp_pieces[r, c] = 1
for r, c in O_moves: my_pieces[r, c] = 1

valid_mask = ((my_pieces + opp_pieces) == 0).flatten()

state = np.zeros((2, 9, 9), dtype=np.float32)
state[0] = my_pieces
state[1] = opp_pieces

print("\n--- Trace Step 4 (Fork Block) ---")
# See if (6, 4) triggers wins_next >= 2
test_action = 6 * 9 + 4
opp_pieces[6, 4] = 1
wins_next = agent._find_winning_actions(opp_pieces, my_pieces)
opp_pieces[6, 4] = 0

print(f"If opp plays (6, 4), wins_next = {wins_next}")
for w in wins_next: print("Opp wins at:", divmod(w, 9))

action = agent.select_action(state, valid_mask, greedy=True)
r, c = divmod(action, 9)
print(f"Action chosen by AI select_action: ({r}, {c})")
