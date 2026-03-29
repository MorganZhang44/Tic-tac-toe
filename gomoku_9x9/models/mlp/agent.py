"""
DQN Agent for 4x4 Connect-4
==============================
Implements Double DQN with:
  - Epsilon-greedy exploration (decaying)
  - Target network (soft updates)
  - Mini-batch training from replay buffer
  - Invalid action masking
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import DuelingConnectNet
from ..common.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Double DQN agent.

    Args:
        board_size    : Side length of the board (4 for core task).
        action_size   : Total number of board cells (16 for 4x4).
        lr            : Learning rate.
        gamma         : Discount factor.
        epsilon_start : Initial exploration rate.
        epsilon_end   : Minimum exploration rate.
        epsilon_decay : Multiplicative decay applied each step.
        batch_size    : Training mini-batch size.
        buffer_size   : Replay buffer capacity.
        target_update : Number of training steps between target-net updates.
        device        : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        board_size: int = 4,
        action_size: int = 16,
        lr: float = 5e-5,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        tau: float = 0.005,
        device: str = None,
    ):
        self.board_size = board_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.train_step = 0

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.policy_net = DuelingConnectNet(board_size, action_size).to(self.device)
        self.target_net = DuelingConnectNet(board_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiser
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss prevents Q-value explosion


        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, valid_mask: np.ndarray,
                      greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection with invalid-move masking.

        Args:
            state      : np.array (2, 4, 4) — current board state
            valid_mask : np.bool array (16,) — True where move is legal
            greedy     : If True, always pick the best Q-value (no explore)

        Returns:
            action (int)
        """
        if not greedy and np.random.rand() < self.epsilon:
            # Random valid move
            valid_actions = np.where(valid_mask)[0]
            return int(np.random.choice(valid_actions))

        # Greedy: pick highest Q-value among valid moves
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        # Mask invalid moves with large negative value
        q_values[~valid_mask] = -1e9
        return int(np.argmax(q_values))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def push(self, state, action, reward, next_state, done):
        """Store a transition AND its horizontal reflection in the replay buffer."""
        # Original transition
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Symmetrical transition (horizontal flip)
        # state shape is (2, 4, 4). axis 2 is the columns.
        state_sym = np.flip(state, axis=2).copy()
        next_state_sym = np.flip(next_state, axis=2).copy()
        
        row, col = divmod(action, self.board_size)
        col_sym = self.board_size - 1 - col
        action_sym = row * self.board_size + col_sym
        
        self.replay_buffer.push(state_sym, action_sym, reward, next_state_sym, done)

    def learn(self) -> float:
        """
        Sample a mini-batch and perform one gradient update.

        Returns:
            loss value (float), or 0.0 if buffer not ready.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for chosen actions
        q_current = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Double DQN target:
        #   action* = argmax_a Q_policy(s', a) [MASKING INVALID MOVES]
        #   target  = r + gamma * Q_target(s', action*) * (1 - done)
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_t)
            
            # Reconstruct valid mask directly from the board state memory
            # state shape is (batch, 2, 4, 4). Any cell where player 1 or player 2 has a piece is invalid.
            occupied = (next_states_t[:, 0, :, :] + next_states_t[:, 1, :, :]).view(-1, self.board_size * self.board_size)
            invalid_mask_t = occupied > 0.5
            
            # Mask so max() ignores illegal moves
            next_q_policy[invalid_mask_t] = -1e9
            
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # Tighter gradient clipping
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Soft update target network (Polyak Averaging)
        self.train_step += 1
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "train_step": self.train_step,
        }, path)
        print(f"[Agent] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt.get("epsilon", self.epsilon_end)
        self.train_step = ckpt.get("train_step", 0)
        self.target_net.eval()
        print(f"[Agent] Loaded checkpoint ← {path}")

    # ------------------------------------------------------------------
    # Public predict API (coursework requirement)
    # ------------------------------------------------------------------

    def predict(self, board_state: np.ndarray):
        """
        Coursework API: given a raw board state, return the best move.

        Args:
            board_state : np.array of shape (4, 4) with values in {0, 1, -1}
                          OR shape (2, 4, 4) already encoded.

        Returns:
            (row, col) : tuple of ints
        """
        # Encode if raw board
        if board_state.ndim == 2:
            state = np.zeros((2, 4, 4), dtype=np.float32)
            # We assume it's our turn (player 1 perspective)
            state[0] = (board_state == 1).astype(np.float32)
            state[1] = (board_state == -1).astype(np.float32)
        else:
            state = board_state.astype(np.float32)

        # Valid moves: any cell that's 0 in the raw board
        if board_state.ndim == 2:
            valid_mask = (board_state.flatten() == 0)
        else:
            # Sum of both channels == 0 → empty cell
            occupied = (state[0] + state[1]).flatten()
            valid_mask = (occupied == 0)

        action = self.select_action(state, valid_mask, greedy=True)
        row, col = divmod(action, self.board_size)
        return int(row), int(col)
