import os
import torch
import numpy as np
import sys

# Ensure we can import from the gomoku_9x9 folder if running from root
sys.path.insert(0, os.path.dirname(__file__))

from models.common.env import Gomoku9x9
from models.cnn_standard.agent import GomokuDQNAgent as StandardAgent
from models.cnn_resnet.agent import GomokuDQNAgent as ResNetAgent
from models.mlp.agent import DQNAgent as MLPAgent

def verify_model(name, agent_class, weights_rel_path, **kwargs):
    print(f"\n[Verification] Testing {name}...")
    try:
        agent = agent_class(**kwargs)
        weights_path = os.path.join(os.path.dirname(__file__), weights_rel_path)
        if os.path.exists(weights_path):
            agent.load(weights_path)
            print(f"  ✓ {name} Weights loaded successfully.")
        else:
            print(f"  ! {name} Weights not found (expected at {weights_rel_path})")
            return False
        
        # Test prediction
        board = np.zeros((9, 9))
        move = agent.predict(board)
        print(f"  ✓ {name} Prediction test passed: move={move}")
        return True
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        # import traceback
        # traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Gomoku 9x9 Model Verification (Restructured) ===")
    
    results = []
    
    # 1. Standard CNN (85MB)
    results.append(verify_model(
        "CNN Standard (85MB)", 
        StandardAgent, 
        "models/cnn_standard/weights.pth",
        board_size=9, action_size=81
    ))
    
    # 2. ResNet CNN (28MB)
    results.append(verify_model(
        "CNN ResNet (28MB)", 
        ResNetAgent, 
        "models/cnn_resnet/weights.pth",
        board_size=9, action_size=81, channels=128, num_res_blocks=6
    ))
    
    # 3. MLP Legacy (No weights for 9x9)
    # results.append(verify_model(
    #     "MLP Legacy", 
    #     MLPAgent, 
    #     "models/mlp/weights.pth",
    #     board_size=9, action_size=81
    # ))
    
    if all(results):
        print("\n🎉 ALL SEARCHED MODELS VERIFIED SUCCESSFULLY!")
    else:
        print("\n⚠️ SOME MODELS FAILED VERIFICATION.")
        sys.exit(1)
