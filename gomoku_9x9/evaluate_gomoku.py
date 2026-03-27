"""
Evaluation Script — 9x9 Gomoku DQN Agent
==========================================
Evaluates the trained Gomoku agent against:
  1. Random agent (should win >80%)
  2. Itself (should be ~50/50)

Usage:
    python evaluate_gomoku.py
    python evaluate_gomoku.py --games 500
    python evaluate_gomoku.py --weights weights/model_weights.pth
"""

import argparse
import os
import numpy as np

from core.gomoku9x9 import Gomoku9x9, RandomAgent
from core.dqn_agent_gomoku import GomokuDQNAgent


def run_match(agent_a: GomokuDQNAgent, agent_b, n_games: int,
              player_a: int = 1) -> dict:
    env      = Gomoku9x9()
    player_b = -player_a
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()
        done  = False
        while not done:
            if env.current_player == player_a:
                mask   = env.get_valid_mask()
                s      = env.get_state_for_player(player_a)
                action = agent_a.select_action(s, mask, greedy=True)
            else:
                if isinstance(agent_b, GomokuDQNAgent):
                    mask   = env.get_valid_mask()
                    s      = env.get_state_for_player(player_b)
                    action = agent_b.select_action(s, mask, greedy=True)
                else:
                    action = agent_b.select_action(env)
            _, _, done, info = env.step(action)

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
        "win_rate":  wins  / n_games,
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
    print(f"  COMP0215 — Gomoku 9x9 DQN Agent Evaluation")
    print(f"{'='*55}")

    if not os.path.exists(args.weights):
        print(f"\n[ERROR] Weights not found at '{args.weights}'")
        print("  → Run 'python train_gomoku.py' first.\n")
        return

    agent = GomokuDQNAgent()
    agent.load(args.weights)
    agent.policy_net.eval()

    n = args.games

    # 1) Agent (X) vs Random (O)
    res1 = run_match(agent, RandomAgent(), n_games=n, player_a=1)
    print_results(f"Agent (X) vs Random (O) — {n} games", res1)

    # 2) Agent (O) vs Random (X)
    res2 = run_match(agent, RandomAgent(), n_games=n, player_a=-1)
    print_results(f"Agent (O) vs Random (X) — {n} games", res2)

    # Combined
    total    = 2 * n
    tw = res1["wins"]   + res2["wins"]
    td = res1["draws"]  + res2["draws"]
    tl = res1["losses"] + res2["losses"]
    # Print total stats
    print(f"\n  ── Combined (both sides) ──")
    print(f"     Wins   : {tw}/{total}  ({tw/total:.1%})")
    print(f"     Draws  : {td}/{total}  ({td/total:.1%})")
    print(f"     Losses : {tl}/{total}  ({tl/total:.1%})")

    # 3) vs Itself
    agent_copy = GomokuDQNAgent()
    agent_copy.load(args.weights)
    agent_copy.policy_net.eval()
    res3 = run_match(agent, agent_copy, n_games=n // 2, player_a=1)
    print_results(f"Agent vs Itself — {n//2} games", res3)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gomoku DQN agent")
    parser.add_argument("--games",   type=int, default=500)
    parser.add_argument("--weights", type=str,
                        default=os.path.join("weights", "model_weights.pth"))
    main(parser.parse_args())
