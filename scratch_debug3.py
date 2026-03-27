import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gomoku.dqn_agent_gomoku import GomokuDQNAgent

agent = GomokuDQNAgent(device="cpu")

my_pieces = np.zeros((9, 9), dtype=np.float32)
my_pieces[5, 1] = 1 # O's blocking piece

opp_pieces = np.zeros((9, 9), dtype=np.float32)
opp_pieces[5, 2] = 1
opp_pieces[5, 3] = 1
opp_pieces[5, 4] = 1
opp_pieces[5, 5] = 1
# Human has 4-in-a-row at 2,3,4,5. Col 6 is EMPTY.

valid_mask = ((my_pieces + opp_pieces) == 0).flatten()

print("Board valid_mask at (5, 6):", valid_mask[5 * 9 + 6])

state = np.zeros((2, 9, 9), dtype=np.float32)
state[0] = my_pieces
state[1] = opp_pieces

print("\n--- Let's trace Step 2 logic directly ---")
opp_wins = agent._find_winning_actions(opp_pieces, my_pieces)
print("opp_wins returned:", opp_wins)
if len(opp_wins) > 0:
    r, c = divmod(opp_wins[0], 9)
    print(f"Agent wants to play at: ({r}, {c})")
else:
    print("Agent failed to find the winning action!")

action = agent.select_action(state, valid_mask, greedy=True)
r, c = divmod(action, 9)
print(f"Action chosen by AI select_action: {r}, {c}")
