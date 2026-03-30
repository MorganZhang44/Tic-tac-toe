# Project Demo Video Script (Part 1: Methodology)

## Section 1: Tic-tac-toe 4x4 (Duration: 50s)
**Visual Focus**: [ttt_dueling_dqn_ultra_simple_1774871673119.png]

*   **0:00 - 0:15**: This 4x4 Tic-tac-toe project serves as the foundation of our research. On such a compact board, we aim not just for high win rates, but for a deep internal understanding of board states. To achieve this, we implemented a **Double Dueling DQN** architecture.
*   **0:15 - 0:35**: The core innovation of the **Dueling network** is the decoupling of the **State Value stream (V)** from the **Action Advantage stream (A)**. This allows the agent to perceive the overall quality of a state, even when no single action is definitively better than others.
*   **0:35 - 0:50**: Combined with **Experience Replay** and **Target Networks**, our training proved highly stable, ultimately achieving a 100% win rate against random baselines and reaching a solid convergence.

---

## Section 2: 28MB ResNet Lite (Duration: 30s)
**Visual Focus**: [resnet_ultra_simple_arch_1774871689603.png]

*   **0:50 - 1:05**: Transitioning to the 9x9 Gomoku challenge, the state space grows exponentially. We introduced this **ResNet (Residual Network)** model. Despite its compact size of only 28MB, it possesses formidable feature extraction capabilities.
*   **1:05 - 1:20**: Featuring **6 Residual Blocks** with **Skip Connections**, the model bypasses traditional depth-related gradient issues. This allows the agent to capture intricate spatial patterns on the 9x9 board while maintaining extreme inference efficiency.

---

## Section 3: 85MB Standard CNN (Duration: 30s)
**Visual Focus**: [standard_cnn_ultra_simple_arch_1774871705407.png]

*   **1:20 - 1:35**: For maximum tactical depth, we developed the **85MB Standard CNN**. This is our "heavyweight" model, prioritizing pure depth and width to suppress opponents through high-capacity pattern recognition.
*   **1:35 - 1:50**: With **128 feature channels per layer**, it processes the 9x9 grid in high-dimensional space. This makes its judgment significantly more robust than lighter models when navigating complex "connect-three" or "open-four" tactics.
