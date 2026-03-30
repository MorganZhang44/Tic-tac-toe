# Project Demo Video Script 
## Section 1: Tic-tac-toe 4x4 (Duration: 50s)
This 4x4 Tic-tac-toe project serves as the foundation of our research. On such a compact board, we aim not just for high win rates, but for a deep internal understanding of board states. To achieve this, we implemented a Double Dueling DQN architecture.

The core innovation of the Dueling network is the decoupling of the State Value stream (V) from the Action Advantage stream (A). This allows the agent to perceive the overall quality of a state, even when no single action is definitively better than others.

Combined with Experience Replay and Target Networks, our training proved highly stable, ultimately achieving a 100% win rate against random baselines and reaching a solid convergence.

---

## Section 2: 28MB ResNet Lite (Duration: 30s)
Transitioning to the 9x9 Gomoku challenge, the state space grows exponentially. We introduced this ResNet (Residual Network) model. Despite its compact size of only 28MB, it possesses formidable feature extraction capabilities.

Featuring 6 Residual Blocks with Skip Connections, the model bypasses traditional depth-related gradient issues. This allows the agent to capture intricate spatial patterns on the 9x9 board while maintaining extreme inference efficiency.

---

## Section 3: 85MB Standard CNN (Duration: 30s)
For maximum tactical depth, we developed the 85MB Standard CNN. This is our "heavyweight" model, prioritizing pure depth and width to suppress opponents through high-capacity pattern recognition.

With 128 feature channels per layer, it processes the 9x9 grid in high-dimensional space. This makes its judgment significantly more robust than lighter models when navigating complex "connect-three" or "open-four" tactics.
