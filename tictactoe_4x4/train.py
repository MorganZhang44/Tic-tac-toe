"""
Training Script — Self-Play DQN for 4x4 Connect-4
====================================================
Two DQN agents (or one agent playing both sides) train via self-play.

Usage:
    python train.py                        # full training (10,000 episodes)
    python train.py --episodes 200         # custom episode count
    python train.py --smoke-test           # 100 episodes, quick sanity check
    python train.py --render               # show board during play (slow)
    python train.py --resume               # continue from existing checkpoint
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from game.tictactoe4x4 import TicTacToe4x4, RandomAgent, SmartAgent
from agent.dqn_agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def evaluate_vs_random(agent: DQNAgent, n_games: int = 200,
                       agent_plays_as: int = 1) -> dict:
    """
    Quick evaluation: agent (greedy) vs. random agent.
    Returns dict with win/draw/loss counts and win_rate.
    """
    env = TicTacToe4x4()
    random_agent = RandomAgent()
    wins = draws = losses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == agent_plays_as:
                mask = env.get_valid_mask()
                action = agent.select_action(state, mask, greedy=True)
            else:
                action = random_agent.select_action(env)

            state, reward, done, info = env.step(action)

        winner = info["winner"]
        if winner == agent_plays_as:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return {
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate": wins / n_games,
        "draw_rate": draws / n_games,
    }


def play_episode(env: TicTacToe4x4,
                 agent1: DQNAgent,
                 agent2: DQNAgent,
                 render: bool = False,
                 learn: bool = True) -> dict:
    """
    Play one self-play episode.

    Both agents observe the board from their own perspective.
    Transitions are stored in each agent's replay buffer by waiting for the 
    opponent's response before forming the (s, a, r, s', done) tuple.
    """
    state = env.reset()
    done = False
    step = 0
    loss1_total = loss2_total = 0.0

    # Bookkeeping: store last state and action for the player that just moved
    prev = {1: None, -1: None}  # {player: (state, action)}

    agent1_player = 1 if np.random.rand() < 0.5 else -1

    while not done:
        player = env.current_player
        agent = agent1 if player == agent1_player else agent2
        mask = env.get_valid_mask()

        # State from the current player's perspective
        s = env.get_state_for_player(player)
        
        if hasattr(agent, "epsilon"):
            action = agent.select_action(s, mask)
        else:
            action = agent.select_action(env)

        if render:
            env.render()

        _, reward, done, info = env.step(action)

        if done:
            if learn and agent is agent1:
                # The current player's episode ends here
                agent1.push(s, action, reward, env.get_state_for_player(player), True)
            if prev[-player] is not None:
                # The OTHER player's episode also ends.
                prev_s, prev_a = prev[-player]
                other_agent = agent2 if player == agent1_player else agent1
                other_reward = -1.0 if reward == 1.0 else 0.0
                if learn and other_agent is agent1:
                    agent1.push(prev_s, prev_a, other_reward, env.get_state_for_player(-player), True)
        else:
            if prev[-player] is not None:
                # The opponent has survived this player's move. THEIR next state is the current board.
                prev_s, prev_a = prev[-player]
                other_agent = agent2 if player == agent1_player else agent1
                if learn and other_agent is agent1:
                    agent1.push(prev_s, prev_a, 0.0, env.get_state_for_player(-player), False)
            
            # Save this player's state and action to await the opponent's response
            prev[player] = (s, action)

        if learn and player == agent1_player:
            loss1_total += agent1.learn()
        
        # Agent 2 does not learn in this architecture (either Frozen or Random)

        step += 1

    return {
        "winner": info["winner"],
        "steps": step,
        "loss1": loss1_total / max(step, 1),
        "loss2": 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'='*60}")
    print(f"  COMP0215 RL Coursework — Connect-4 4x4 DQN Training")
    print(f"{'='*60}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Device    : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  Smoke test: {args.smoke_test}")
    print(f"{'='*60}\n")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = TicTacToe4x4()

    # Create two agents that share the same replay buffer style
    agent1 = DQNAgent()
    # Agent 2 is the same class but independent weights (self-play)
    agent2 = DQNAgent()

    # Resume from checkpoint if requested
    weights_path = os.path.join("weights", "model_weights.pth")
    if args.resume and os.path.exists(weights_path):
        agent1.load(weights_path)
        print("[Train] Resumed agent1 from checkpoint.")
    elif not args.resume:
        print("[Train] Starting fresh training (use --resume to continue from checkpoint).")

    # CSV logging — overwrite unless resuming
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "training_log.csv")
    if args.resume and os.path.exists(log_path):
        csv_file = open(log_path, "a", newline="")
        writer = csv.writer(csv_file)
        print(f"[Train] Appending to existing log: {log_path}")
    else:
        csv_file = open(log_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "winner", "steps",
                         "loss1", "loss2",
                         "epsilon",
                         "win_rate_vs_random", "draw_rate_vs_random"])

    random_agent = RandomAgent()
    smart_agent = SmartAgent()

    best_win_rate = 0.0
    eval_interval = max(50, args.episodes // 100)
    log_interval  = max(10, args.episodes // 200)
    freeze_interval = max(100, args.episodes // 20) # Update opponent every 5% of training
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        # Update frozen opponent periodically
        if ep % freeze_interval == 0:
            agent2.policy_net.load_state_dict(agent1.policy_net.state_dict())
            agent2.target_net.load_state_dict(agent1.target_net.state_dict())
            print(f"  [Train] Updated frozen opponent at episode {ep}")
    
        # 33% Random Agent, 33% Smart Agent, 34% Frozen Agent
        # Forces the DQN to beat random noise, respect 1-step threats, and defeat its own logic.
        r = random.random()
        if r < 0.33:
            opponent = random_agent
        elif r < 0.66:
            opponent = smart_agent
        else:
            opponent = agent2

        result = play_episode(env, agent1, opponent,
                              render=args.render, learn=True)

        if ep % log_interval == 0:
            elapsed = time.time() - start_time

            # Quick evaluation every eval_interval episodes
            win_rate = draw_rate = None
            if ep % eval_interval == 0:
                eval_res = evaluate_vs_random(agent1, n_games=100)
                win_rate  = eval_res["win_rate"]
                draw_rate = eval_res["draw_rate"]

                # Save best model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    agent1.save(weights_path)

                print(f"  Ep {ep:>6}/{args.episodes} | "
                      f"ε={agent1.epsilon:.3f} | "
                      f"loss={result['loss1']:.4f} | "
                      f"win_vs_random={win_rate:.2%} | "
                      f"best={best_win_rate:.2%} | "
                      f"elapsed={elapsed:.0f}s")

            writer.writerow([
                ep, result["winner"], result["steps"],
                f"{result['loss1']:.6f}", f"{result['loss2']:.6f}",
                f"{agent1.epsilon:.4f}",
                f"{win_rate:.4f}" if win_rate is not None else "",
                f"{draw_rate:.4f}" if draw_rate is not None else "",
            ])
            csv_file.flush()

    csv_file.close()

    # Final save
    agent1.save(weights_path)
    print(f"\n[Train] Done! Best win-rate vs random: {best_win_rate:.2%}")
    print(f"[Train] Weights → {weights_path}")
    print(f"[Train] Log     → {log_path}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN agent for 4x4 Connect-4")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Number of training episodes (default: 10000)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only 100 episodes for a quick sanity check")
    parser.add_argument("--render", action="store_true",
                        help="Render board during training (slow)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.smoke_test:
        args.episodes = 100

    train(args)
