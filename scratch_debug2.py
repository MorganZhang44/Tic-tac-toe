import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gomoku.dqn_agent_gomoku import GomokuDQNAgent

agent = GomokuDQNAgent(device="cpu")

# Simple board: Player O is AI (my_pieces), Player X is Human (opp_pieces)
# Human has pieces at (4,3), (4,4), (4,5) -> This is an open 3!
# AI has piece at (0,0)

my_pieces = np.zeros((9, 9), dtype=np.float32)
my_pieces[0, 0] = 1

opp_pieces = np.zeros((9, 9), dtype=np.float32)
opp_pieces[4, 3] = 1
opp_pieces[4, 4] = 1
opp_pieces[4, 5] = 1

valid_mask = ((my_pieces + opp_pieces) == 0).flatten()

state = np.zeros((2, 9, 9), dtype=np.float32)
state[0] = my_pieces
state[1] = opp_pieces

print("Open 3 on board at (4,3), (4,4), (4,5).")
print("AI pieces (my_pieces):\n", my_pieces)
print("Human pieces (opp_pieces):\n", opp_pieces)

action = agent.select_action(state, valid_mask, greedy=True)
r, c = divmod(action, 9)
print(f"Action chosen by AI: {r}, {c}")

if (r, c) in [(4, 2), (4, 6)]:
    print("SUCCESS: AI properly blocked the open 3!")
else:
    print("FAILURE: AI did NOT block the open 3.")
    
    print("\n--- Let's trace Step 4 logic ---")
    warning_blocks = []
    for test_action in np.where(valid_mask)[0]:
        tr, tc = divmod(test_action, 9)
        opp_pieces[tr, tc] = 1
        wins_next = agent._find_winning_actions(opp_pieces, my_pieces)
        opp_pieces[tr, tc] = 0
        
        if tr == 4 and tc == 2:
            print(f"When simulating opponent playing at (4, 2):")
            print(f"wins_next = {wins_next}")
            
        if len(wins_next) >= 2:
            warning_blocks.append((tr, tc))
            
    print("Warning blocks found by Step 4:", warning_blocks)
