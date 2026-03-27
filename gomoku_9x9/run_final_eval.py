import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from evaluate_gomoku import run_match
from gomoku.dqn_agent_gomoku import GomokuDQNAgent
from gomoku.gomoku9x9 import RandomAgent, SmartAgent

def main():
    print("Loading V3 Hybrid Adversarial Agent...")
    agent = GomokuDQNAgent(device="cpu") # Eval is fast enough on CPU
    weight_path = os.path.join("weights", "gomoku_weights.pth")
    agent.load(weight_path)
    agent.policy_net.eval()
    
    n_games = 5
    print(f"Evaluating vs RandomAgent ({n_games} games)...")
    res_random_x = run_match(agent, RandomAgent(), n_games, player_a=1)
    res_random_o = run_match(agent, RandomAgent(), n_games, player_a=-1)
    
    win_rate_random = (res_random_x["wins"] + res_random_o["wins"]) / (2 * n_games)
    draw_rate_random = (res_random_x["draws"] + res_random_o["draws"]) / (2 * n_games)
    loss_rate_random = (res_random_x["losses"] + res_random_o["losses"]) / (2 * n_games)

    print(f"Evaluating vs SmartAgent ({n_games} games)...")
    res_smart_x = run_match(agent, SmartAgent(), n_games, player_a=1)
    res_smart_o = run_match(agent, SmartAgent(), n_games, player_a=-1)
    
    win_rate_smart = (res_smart_x["wins"] + res_smart_o["wins"]) / (2 * n_games)
    draw_rate_smart = (res_smart_x["draws"] + res_smart_o["draws"]) / (2 * n_games)
    loss_rate_smart = (res_smart_x["losses"] + res_smart_o["losses"]) / (2 * n_games)
    
    print("\n--- Final Statistics ---")
    print(f"Vs Random: {win_rate_random:.1%} Win | {draw_rate_random:.1%} Draw | {loss_rate_random:.1%} Loss")
    print(f"Vs Smart : {win_rate_smart:.1%} Win | {draw_rate_smart:.1%} Draw | {loss_rate_smart:.1%} Loss")
    
    # Plotting
    labels = ['Vs Random Agent', 'Vs Smart Agent']
    win_rates = [win_rate_random, win_rate_smart]
    draw_rates = [draw_rate_random, draw_rate_smart]
    loss_rates = [loss_rate_random, loss_rate_smart]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors
    color_win = '#2ecc71'
    color_draw = '#f1c40f'
    color_loss = '#e74c3c'
    
    rects1 = ax.bar(x - width, win_rates, width, label='Win', color=color_win, edgecolor='black')
    rects2 = ax.bar(x, draw_rates, width, label='Draw', color=color_draw, edgecolor='black')
    rects3 = ax.bar(x + width, loss_rates, width, label='Loss', color=color_loss, edgecolor='black')
    
    # Add text
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Final Gomoku AI V3 Benchmark Results (20 Games Sample)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=11)
    
    # Annotate bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate('{:.1%}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    
    # Save to artifacts
    out_path = r"C:\Users\Morgan\.gemini\antigravity\brain\394da746-2a01-4423-8959-501d403b10d5\final_gomoku_benchmark.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to: {out_path}")
    
if __name__ == "__main__":
    main()
