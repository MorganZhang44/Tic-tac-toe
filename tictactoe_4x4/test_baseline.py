from game.tictactoe4x4 import TicTacToe4x4, RandomAgent
env = TicTacToe4x4()
agent1 = RandomAgent()
agent2 = RandomAgent()
wins, draws, losses = 0, 0, 0
for _ in range(10000):
   env.reset()
   done = False
   while not done:
       mask = env.get_valid_mask()
       action = agent1.select_action(env) if env.current_player == 1 else agent2.select_action(env)
       _, _, done, info = env.step(action)
   if info["winner"] == 1: wins += 1
   elif info["winner"] == -1: losses += 1
   else: draws += 1
print(f"Random vs Random: P1 wins {wins/100}% | Draws {draws/100}% | P2 wins {losses/100}%")
