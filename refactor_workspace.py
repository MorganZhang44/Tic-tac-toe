import os
import shutil
import glob

def replace_in_file(filepath, old_text, new_text):
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_text in content:
        content = content.replace(old_text, new_text)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    base_dir = r"C:\Users\Morgan\Desktop\Tic-tac-toe"
    os.chdir(base_dir)

    # Create target directories
    os.makedirs("tictactoe_4x4", exist_ok=True)
    os.makedirs("gomoku_9x9", exist_ok=True)
    os.makedirs("tictactoe_4x4/logs", exist_ok=True)
    os.makedirs("tictactoe_4x4/weights", exist_ok=True)
    os.makedirs("tictactoe_4x4/tests", exist_ok=True)
    os.makedirs("gomoku_9x9/logs", exist_ok=True)
    os.makedirs("gomoku_9x9/weights", exist_ok=True)
    os.makedirs("gomoku_9x9/tests", exist_ok=True)

    # 1. Move Tic-Tac-Toe Files
    tictactoe_files = [
        "train.py", "play.py", "evaluate.py", "predict.py", 
        "test_baseline.py", "plot_results.py"
    ]
    for f in tictactoe_files:
        if os.path.exists(f): shutil.move(f, os.path.join("tictactoe_4x4", f))
    
    if os.path.exists("game"): shutil.move("game", "tictactoe_4x4/game")
    if os.path.exists("agent"): shutil.move("agent", "tictactoe_4x4/agent")
    
    if os.path.exists("tests/test_game.py"):
        shutil.move("tests/test_game.py", "tictactoe_4x4/tests/test_game.py")

    # TicTacToe Logs & Weights
    tt_logs = ["training_log.csv", "training_curves.png", "eval_output.txt", "full_training_output.txt"]
    for l in tt_logs:
        src = os.path.join("logs", l)
        if os.path.exists(src): shutil.move(src, os.path.join("tictactoe_4x4/logs", l))
        
    tt_weights = ["model_weights.pth"]
    for w in tt_weights:
        src = os.path.join("weights", w)
        if os.path.exists(src): shutil.move(src, os.path.join("tictactoe_4x4/weights", w))

    # 2. Move Gomoku Files
    gomoku_files = [
        "train_gomoku.py", "play_gomoku.py", "evaluate_gomoku.py", 
        "predict_gomoku.py", "plot_gomoku_results.py", "run_final_eval.py"
    ]
    for f in gomoku_files:
        if os.path.exists(f): shutil.move(f, os.path.join("gomoku_9x9", f))

    # Rename the gomoku module to core to avoid import clashes
    if os.path.exists("gomoku"): shutil.move("gomoku", "gomoku_9x9/core")
    
    # We need Gomoku to have the agent/replay_buffer.py! Since Gomoku used `agent.replay_buffer`,
    # we must copy the generic `agent` folder to gomoku_9x9 as well to keep them independent.
    if os.path.exists("tictactoe_4x4/agent"):
        shutil.copytree("tictactoe_4x4/agent", "gomoku_9x9/agent", dirs_exist_ok=True)

    if os.path.exists("tests/test_gomoku.py"):
        shutil.move("tests/test_gomoku.py", "gomoku_9x9/tests/test_gomoku.py")

    # Gomoku Logs & Weights
    go_logs = ["gomoku_training_log.csv", "gomoku_training_curves.png"]
    for l in go_logs:
        src = os.path.join("logs", l)
        if os.path.exists(src): shutil.move(src, os.path.join("gomoku_9x9/logs", l))
        
    go_weights = ["gomoku_weights.pth"]
    for w in go_weights:
        src = os.path.join("weights", w)
        if os.path.exists(src): shutil.move(src, os.path.join("gomoku_9x9/weights", w))

    # Clean up empty old folders
    for d in ["tests", "logs", "weights"]:
        if os.path.exists(d) and not os.listdir(d):
            os.rmdir(d)

    # 3. Hot-patch Gomoku Imports
    # The module was 'gomoku', now it's 'core' inside 'gomoku_9x9'
    gomoku_py_files = glob.glob("gomoku_9x9/**/*.py", recursive=True)
    for filepath in gomoku_py_files:
        replace_in_file(filepath, "from gomoku.", "from core.")
        replace_in_file(filepath, "import gomoku.", "import core.")
        # Any string paths
        replace_in_file(filepath, "logs/gomoku_training_log.csv", "logs/training_log.csv")
        replace_in_file(filepath, "weights/gomoku_weights.pth", "weights/model_weights.pth")
        replace_in_file(filepath, "logs/gomoku_training_curves.png", "logs/training_curves.png")
        
    # Rename the gomoku log/weight files to generic names inside their isolated folder
    if os.path.exists("gomoku_9x9/logs/gomoku_training_log.csv"):
        os.rename("gomoku_9x9/logs/gomoku_training_log.csv", "gomoku_9x9/logs/training_log.csv")
    if os.path.exists("gomoku_9x9/weights/gomoku_weights.pth"):
        os.rename("gomoku_9x9/weights/gomoku_weights.pth", "gomoku_9x9/weights/model_weights.pth")
    if os.path.exists("gomoku_9x9/logs/gomoku_training_curves.png"):
        os.rename("gomoku_9x9/logs/gomoku_training_curves.png", "gomoku_9x9/logs/training_curves.png")
        
    # Also fix imports inside gomoku_9x9/core/dqn_agent_gomoku.py
    # From: from gomoku.network_gomoku import GomokuDuelingNet
    # To:   from core.network_gomoku import GomokuDuelingNet
    
    print("Refactoring complete! The projects are now modularized.")

if __name__ == "__main__":
    main()
