"""
Plot training curves from logs/training_log.csv.
Generates PNG plots saved to logs/.

Usage:
    python plot_results.py
"""

import os
import sys
import csv
import numpy as np


def load_log(path="logs/training_log.csv"):
    episodes, epsilons, losses, win_rates = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            epsilons.append(float(row["epsilon"]))
            losses.append(float(row["loss1"]) if row["loss1"] else None)
            win_rates.append(float(row["win_rate_vs_random"])
                             if row["win_rate_vs_random"] else None)
    return episodes, epsilons, losses, win_rates


def smooth(values, window=20):
    """Simple moving average."""
    result = []
    for i, v in enumerate(values):
        if v is None:
            result.append(None)
            continue
        start = max(0, i - window)
        chunk = [x for x in values[start:i+1] if x is not None]
        result.append(np.mean(chunk) if chunk else None)
    return result


def main():
    log_path = "logs/training_log.csv"
    if not os.path.exists(log_path):
        print(f"[ERROR] Log file not found: {log_path}")
        print("  → Run train.py first, then re-run this script.")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not found. Install it with:\n"
              "  pip install matplotlib")
        sys.exit(1)

    episodes, epsilons, losses, win_rates = load_log(log_path)
    s_losses   = smooth(losses, 30)
    s_win_rates = smooth(win_rates, 5)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("DQN Training — 4×4 Connect-4", fontsize=14, fontweight="bold")

    # ── Loss ──
    ax = axes[0]
    valid_ep = [e for e, l in zip(episodes, s_losses) if l is not None]
    valid_l  = [l for l in s_losses if l is not None]
    ax.plot(valid_ep, valid_l, color="#4C72B0", linewidth=1.5)
    ax.set_ylabel("Huber Loss (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.3)

    # ── Win Rate ──
    ax = axes[1]
    valid_ep_w = [e for e, w in zip(episodes, s_win_rates) if w is not None]
    valid_w    = [w for w in s_win_rates if w is not None]
    ax.plot(valid_ep_w, valid_w, color="#55A868", linewidth=1.5)
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% target")
    ax.set_ylabel("Win Rate vs Random (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_title("Win Rate vs Random Agent")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Epsilon ──
    ax = axes[2]
    ax.plot(episodes, epsilons, color="#C44E52", linewidth=1.5)
    ax.set_ylabel("Epsilon")
    ax.set_xlabel("Episode")
    ax.set_title("Exploration Rate (ε) Decay")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "logs/training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
