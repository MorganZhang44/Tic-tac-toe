# COMP0215 RL Coursework — Dueling DQN Game Agents

## Overview
This repository contains two reinforcement learning projects built for the COMP0215 coursework:
1. **Tic-tac-toe 4×4**: The foundational project (Double Dueling DQN) trained to play a 4x4 variant of Connect-4.
2. **Gomoku 9×9 (V3 MCTS-Hybrid)**: An immensely upgraded Dueling DQN architecture featuring Convolutional Neural Networks (CNN), Convolutional Spatial Priors, and a 1-ply MCTS (Monte Carlo Tree Search) Tactical Filter. 

Both agents were built completely from scratch using PyTorch and trained via massive autonomous self-play, achieving absolute dominance against Random and Heuristic AI baselines.

---

## Project Structure
The repository has been elegantly decoupled into two isolated modules:

```
Tic-tac-toe/
├── tictactoe_4x4/             # [Project 1] 4x4 Tic-tac-toe
│   ├── agent/                 # Dueling DQN logic
│   ├── game/                  # 4x4 Environment
│   ├── logs/ & weights/       # Old training logs and saved models
│   ├── train.py               # Training entry point
│   ├── play.py                # Human vs AI CLI
│   ├── evaluate.py
│   └── predict.py             # Coursework array-to-tuple API
│
└── gomoku_9x9/                # [Project 2] 9x9 Gomoku (V3 Tactical AI)
    ├── core/                  # Gomoku 9x9 Env, CNN Network, and Hybrid Agent
    ├── agent/                 # Generic uniform replay buffer
    ├── logs/ & weights/       # 30,000+ episode logs and 10-crown AI Brain
    ├── train_gomoku.py        # Curriculum MCTS-infused training script
    ├── play_gomoku.py         # 🎮 GUI Pygame Application (Must Try!)
    ├── evaluate_gomoku.py
    └── predict_gomoku.py      # Coursework array-to-tuple API
```

---

## Setup & Installation
```bash
# We highly recommend using Conda
conda create -n rlcw python=3.10
conda activate rlcw

# Install all dependencies
pip install torch numpy pygame pytest matplotlib
```

---

## 🎮 Interactive Play Modes (GUI)

This repository provides highly interactive Pygame UIs for both games, featuring multiple AI architectures and battle modes.

### 1. Gomoku 9×9 — Three Advanced Modes
Navigate to `gomoku_9x9/` and choose your challenge:

| Mode | Command | Description |
| :--- | :--- | :--- |
| **🥇 Standard CNN** | `python3 play_gomoku.py` | Challenge our strongest 85MB model (Dueling DQN + 4 ResBlocks). |
| **🥈 ResNet Lite** | `python3 play_gomoku.py --resnet` | Faster, lightweight 28MB model with specialized residual skip connections. |
| **⚔️ Battle Mode** | `python3 play_gomoku.py --battle` | **Recommended!** Watch the 85MB Standard CNN vs. 28MB ResNet in a direct duel. |

*   **Options**: Add `--ai_first` to let the AI take the first move (Black), or `--ai_vs_ai` (non-battle) to watch self-play.

### 2. Tic-tac-toe 4×4
Navigate to `tictactoe_4x4/` and run:
*   **Human vs AI**: `python3 play.py`
*   **AI Auto-play**: `python3 play.py --ai-vs-ai`

---

## Algorithm Details (Gomoku V3 Hybrid)
- **Deep Intuition**: A Dueling DQN architecture equipped with a deep CNN backbone and 4 Residual Blocks.
- **Hardware-Agnostic Resilience**: Dynamic compatibility fallbacks that automatically transition CNN operations to pure CPU logic to overcome known Blackwell (RTX 5080) PyTorch compilation bugs.
- **MCTS Expert Iteration**: The training loop runs 1-step Monte Carlo tree checks over all empty board positions in purely simulated Numpy arrays, injecting immensely high-quality expert responses into the Replay Buffer.
- **Absolute Local Proximity**: The agent utilizes 2D spatial convolution masks to organically restrict its own candidate generation space to radius-1 offsets of currently occupied tiles, eliminating edge-wandering AI hallucinations.

---

## Evaluation & Automated Testing

You can evaluate the trained weights against the `RandomAgent` or evaluate the coursework `predict` APIs via:

**For 9x9 Gomoku:**
```bash
cd gomoku_9x9

# 1. Run Coursework API Integrity Check (Smoke test)
python predict_gomoku.py --test

# 2. Evaluate mathematically vs Random Baseline (Generates 100% Win Rate)
python evaluate_gomoku.py --games 500

# 3. Re-print the benchmark Loss / Training curves
python plot_gomoku_results.py
```

**For 4x4 Tic-tac-toe:**
```bash
cd tictactoe_4x4
python evaluate.py --games 500
python predict.py --test
```
