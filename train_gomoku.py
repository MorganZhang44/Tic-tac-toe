"""
Training Script — Self-Play DQN for 9x9 Gomoku
================================================
Trains a DQN agent via a progressive curriculum:
  33% Random opponent | 33% Smart opponent | 34% Frozen self-play

Usage:
    python train_gomoku.py                         # 30,000 episodes
    python train_gomoku.py --episodes 5000         # custom count
    python train_gomoku.py --smoke-test            # 200 episodes
    python train_gomoku.py --resume                # continue from checkpoint
    python train_gomoku.py --render                # show board (slow)
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from gomoku.gomoku9x9 import Gomoku9x9, RandomAgent, SmartAgent
from gomoku.dqn_agent_gomoku import GomokuDQNAgent


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def evaluate_vs_random(agent: GomokuDQNAgent, n_games: int = 100,
                       agent_plays_as: int = 1) -> dict:
    """Agent (greedy) vs. RandomAgent — returns win/draw/loss stats."""
    env = Gomoku9x9()
    random_opp = RandomAgent()
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            if env.current_player == agent_plays_as:
                mask   = env.get_valid_mask()
                s      = env.get_state_for_player(agent_plays_as)
                action = agent.select_action(s, mask, greedy=True)
            else:
                action = random_opp.select_action(env)
            state, _, done, info = env.step(action)

        winner = info["winner"]
        if winner == agent_plays_as:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return {
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate":  wins / n_games,
        "draw_rate": draws / n_games,
    }


def play_episode(env: Gomoku9x9,
                 agent1: GomokuDQNAgent,
                 agent2,
                 render: bool = False,
                 learn: bool = True) -> dict:
    """
    Play one episode. agent1 learns; agent2 is a fixed opponent.
    Transitions use the deferred reward pattern (wait for opponent's response).
    """
    state = env.reset()
    done  = False
    step  = 0
    loss_total = 0.0

    prev = {1: None, -1: None}
    agent1_player = 1 if np.random.rand() < 0.5 else -1

    while not done:
        player = env.current_player
        is_agent1 = (player == agent1_player)
        mask = env.get_valid_mask()
        s    = env.get_state_for_player(player)

        if is_agent1:
            action = agent1.select_action(s, mask)
        else:
            if hasattr(agent2, "select_action") and callable(
                    getattr(agent2, "select_action")):
                if isinstance(agent2, GomokuDQNAgent):
                    action = agent2.select_action(s, mask, greedy=True)
                else:
                    action = agent2.select_action(env)
            else:
                action = agent2.select_action(env)

        if render:
            env.render()

        _, reward, done, info = env.step(action)

        if done:
            if learn and is_agent1:
                agent1.push(s, action, reward,
                            env.get_state_for_player(player), True)
            # Give the OTHER agent the complementary reward
            if prev[-player] is not None:
                prev_s, prev_a = prev[-player]
                other_is_agent1 = (player != agent1_player)
                other_reward = -1.0 if reward == 1.0 else 0.0
                if learn and other_is_agent1:
                    agent1.push(prev_s, prev_a, other_reward,
                                env.get_state_for_player(-player), True)
        else:
            if prev[-player] is not None:
                prev_s, prev_a = prev[-player]
                other_is_agent1 = (player != agent1_player)
                if learn and other_is_agent1:
                    agent1.push(prev_s, prev_a, 0.0,
                                env.get_state_for_player(-player), False)
            prev[player] = (s, action)

        if learn and is_agent1:
            loss_total += agent1.learn()

        step += 1

    return {
        "winner": info["winner"],
        "steps":  step,
        "loss":   loss_total / max(step, 1),
    }


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  COMP0215 RL Coursework - Gomoku 9x9 DQN Training")
    print(f"{'='*65}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"{'='*65}\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env    = Gomoku9x9()
    agent1 = GomokuDQNAgent()
    agent2 = GomokuDQNAgent()   # frozen opponent

    weights_path = os.path.join("weights", "gomoku_weights.pth")
    if args.resume and os.path.exists(weights_path):
        agent1.load(weights_path)
        agent2.policy_net.load_state_dict(
            agent1.policy_net._orig_mod.state_dict()
            if hasattr(agent1.policy_net, "_orig_mod")
            else agent1.policy_net.state_dict()
        )
        print("[Train] Resumed from checkpoint.")
    else:
        print("[Train] Starting fresh training (use --resume to continue).")

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "gomoku_training_log.csv")
    if args.resume and os.path.exists(log_path):
        csv_file = open(log_path, "a", newline="")
        writer   = csv.writer(csv_file)
    else:
        csv_file = open(log_path, "w", newline="")
        writer   = csv.writer(csv_file)
        writer.writerow(["episode", "winner", "steps", "loss", "epsilon",
                         "win_rate_vs_random", "draw_rate_vs_random"])

    random_agent = RandomAgent()
    smart_agent  = SmartAgent()

    best_win_rate   = 0.0
    eval_interval   = max(100, args.episodes // 50)
    log_interval    = max(20,  args.episodes // 200)
    freeze_interval = max(200, args.episodes // 15)

    start_time = time.time()

    for ep in range(1, args.episodes + 1):

        # Update frozen opponent copy
        if ep % freeze_interval == 0:
            src = (agent1.policy_net._orig_mod
                   if hasattr(agent1.policy_net, "_orig_mod")
                   else agent1.policy_net)
            tgt = (agent2.policy_net._orig_mod
                   if hasattr(agent2.policy_net, "_orig_mod")
                   else agent2.policy_net)
            tgt.load_state_dict(src.state_dict())
            agent2.target_net.load_state_dict(tgt.state_dict())
            print(f"  [Train] Updated frozen opponent at episode {ep}")

        # Curriculum: 33% Random, 33% Smart, 34% Frozen self
        r = random.random()
        if r < 0.33:
            opponent = random_agent
        elif r < 0.66:
            opponent = smart_agent
        else:
            opponent = agent2

        result = play_episode(env, agent1, opponent, render=args.render, learn=True)

        if ep % log_interval == 0:
            elapsed = time.time() - start_time
            win_rate = draw_rate = None

            if ep % eval_interval == 0:
                eval_res  = evaluate_vs_random(agent1, n_games=100)
                win_rate  = eval_res["win_rate"]
                draw_rate = eval_res["draw_rate"]

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    agent1.save(weights_path)

                print(f"  Ep {ep:>7}/{args.episodes} | "
                      f"ε={agent1.epsilon:.3f} | "
                      f"loss={result['loss']:.4f} | "
                      f"win_vs_random={win_rate:.2%} | "
                      f"best={best_win_rate:.2%} | "
                      f"elapsed={elapsed:.0f}s")

            writer.writerow([
                ep, result["winner"], result["steps"],
                f"{result['loss']:.6f}",
                f"{agent1.epsilon:.4f}",
                f"{win_rate:.4f}"  if win_rate  is not None else "",
                f"{draw_rate:.4f}" if draw_rate is not None else "",
            ])
            csv_file.flush()

    csv_file.close()
    agent1.save(weights_path)
    print(f"\n[Train] Done! Best win-rate vs random: {best_win_rate:.2%}")
    print(f"[Train] Weights → {weights_path}")
    print(f"[Train] Log     → {log_path}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN agent for 9x9 Gomoku")
    parser.add_argument("--episodes", type=int, default=30_000,
                        help="Number of training episodes (default: 30000)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 200 episodes for a quick sanity check")
    parser.add_argument("--render", action="store_true",
                        help="Render board during training (slow)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.smoke_test:
        args.episodes = 200

    train(args)
