"""
Evaluation Script
==================
Evaluates a trained DQN agent against:
  1. A random agent   (win rate should be very high)
  2. Itself (reverse sides)  (should be roughly 50/50)

Usage:
    python evaluate.py                        # 1000 games vs random
    python evaluate.py --games 500            # custom number of games
    python evaluate.py --weights path/to/w   # custom weights file
"""

import argparse
import numpy as np
import os

from game.tictactoe4x4 import TicTacToe4x4, RandomAgent
from agent.dqn_agent import DQNAgent


def run_match(agent_a: DQNAgent, agent_b, n_games: int,
              player_a: int = 1) -> dict:
    """
    Run n_games between agent_a (as player_a) and agent_b.
    agent_b can be a DQNAgent or a RandomAgent.

    Returns outcome dictionary.
    """
    env = TicTacToe4x4()
    player_b = -player_a
    wins = draws = losses = 0

    for g in range(n_games):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == player_a:
                mask = env.get_valid_mask()
                s = env.get_state_for_player(player_a)
                action = agent_a.select_action(s, mask, greedy=True)
            else:
                if isinstance(agent_b, DQNAgent):
                    mask = env.get_valid_mask()
                    s = env.get_state_for_player(player_b)
                    action = agent_b.select_action(s, mask, greedy=True)
                else:
                    action = agent_b.select_action(env)

            state, reward, done, info = env.step(action)

        winner = info["winner"]
        if winner == player_a:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return {
        "wins":      wins,
        "draws":     draws,
        "losses":    losses,
        "games":     n_games,
        "win_rate":  wins / n_games,
        "draw_rate": draws / n_games,
        "loss_rate": losses / n_games,
    }


def print_results(label: str, res: dict):
    print(f"\n  ── {label} ──")
    print(f"     Games  : {res['games']}")
    print(f"     Wins   : {res['wins']}  ({res['win_rate']:.1%})")
    print(f"     Draws  : {res['draws']}  ({res['draw_rate']:.1%})")
    print(f"     Losses : {res['losses']}  ({res['loss_rate']:.1%})")


def main(args):
    print(f"\n{'='*55}")
    print(f"  COMP0215 — DQN Agent Evaluation")
    print(f"{'='*55}")

    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"\n[ERROR] Weights not found at '{weights_path}'")
        print("  → Run 'python train.py' first to generate weights.\n")
        return

    # Load agent
    agent = DQNAgent()
    agent.load(weights_path)
    agent.policy_net.eval()

    n = args.games

    # 1) vs Random (agent plays as player 1)
    res1 = run_match(agent, RandomAgent(), n_games=n, player_a=1)
    print_results(f"Agent (X) vs Random (O) — {n} games", res1)

    # 2) vs Random (agent plays as player 2)
    res2 = run_match(agent, RandomAgent(), n_games=n, player_a=-1)
    print_results(f"Agent (O) vs Random (X) — {n} games", res2)

    # Combined
    total_games = res1["games"] + res2["games"]
    total_wins  = res1["wins"]  + res2["wins"]
    total_draws = res1["draws"] + res2["draws"]
    total_losses = res1["losses"] + res2["losses"]

    print(f"\n  ── Combined (both sides) ──")
    print(f"     Wins   : {total_wins}/{total_games}  ({total_wins/total_games:.1%})")
    print(f"     Draws  : {total_draws}/{total_games}  ({total_draws/total_games:.1%})")
    print(f"     Losses : {total_losses}/{total_games}  ({total_losses/total_games:.1%})")

    # 3) vs Itself (another loaded copy of the same agent)
    agent_copy = DQNAgent()
    agent_copy.load(weights_path)
    agent_copy.policy_net.eval()

    res3 = run_match(agent, agent_copy, n_games=n//2, player_a=1)
    print_results(f"Agent vs Itself — {n//2} games", res3)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN agent")
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of evaluation games (default: 1000)")
    parser.add_argument("--weights", type=str,
                        default=os.path.join("weights", "model_weights.pth"),
                        help="Path to model weights")
    args = parser.parse_args()
    main(args)
