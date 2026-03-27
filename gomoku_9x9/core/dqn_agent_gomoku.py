"""
DQN Agent for 9x9 Gomoku
=========================
Implements Double DQN with:
  - Dueling CNN network (residual blocks)
  - Epsilon-greedy exploration (decaying)
  - Soft target network updates (Polyak averaging)
  - Mini-batch training from replay buffer
  - Invalid action masking
  - 8-fold data augmentation (4 rotations × 2 flips)
  - Mixed-precision training (torch.cuda.amp) — fast on RTX 5080
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from core.network_gomoku import DuelingGomokuNet

# Reuse the existing replay buffer from the base project
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.replay_buffer import ReplayBuffer


class GomokuDQNAgent:
    """
    Double DQN agent for 9x9 Gomoku.

    Args:
        board_size    : 9 for standard Gomoku
        action_size   : 81 (9×9 board cells)
        lr            : Learning rate
        gamma         : Discount factor
        epsilon_start : Initial exploration rate
        epsilon_end   : Minimum exploration rate
        epsilon_decay : Multiplicative decay per training step
        batch_size    : Mini-batch size (256 recommended for RTX 5080)
        buffer_size   : Replay buffer capacity
        tau           : Soft update coefficient
        channels      : CNN filter count
        num_res_blocks: Number of residual blocks in the network
        device        : 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        board_size: int = 9,
        action_size: int = 81,
        lr: float = 3e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9998,
        batch_size: int = 256,
        buffer_size: int = 200_000,
        tau: float = 0.005,
        channels: int = 128,
        num_res_blocks: int = 6,
        device: str = None,
    ):
        self.board_size  = board_size
        self.action_size = action_size
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_end   = max(0.05, epsilon_end)
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.tau         = tau
        self.train_step    = 0

        # Device selection with auto-fallback for missing kernel images (e.g. RTX 5080)
        if device is None:
            if torch.cuda.is_available():
                try:
                    # Test if the current PyTorch installation actually supports this GPU's architecture
                    _ = torch.zeros(1).cuda() + 1
                    self.device = torch.device("cuda")
                except RuntimeError:
                    print("[GomokuAgent] WARNING: CUDA is available but missing kernels for this GPU architecture (e.g. RTX 5080). Falling back to CPU.")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[GomokuAgent] Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"[GomokuAgent] GPU: {torch.cuda.get_device_name(0)}")

        # Networks
        self.policy_net = DuelingGomokuNet(
            board_size, action_size, channels, num_res_blocks
        ).to(self.device)
        self.target_net = DuelingGomokuNet(
            board_size, action_size, channels, num_res_blocks
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Disable torch.compile on RTX 5080.
        # The RTX 5080 uses Blackwell architecture (sm_120), which is currently too new for 
        # the default PyTorch 2.5 Triton compiler and crashes during the first forward pass.
        # We will use standard eager execution with AMP, which is still incredibly fast.
        
        # Optimiser
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()

        # Mixed precision scaler (A100/RTX-level speedup)
        # Disabled because nightly PyTorch float16 convolutions on Blackwell (sm_120) 
        # sometimes hit missing kernel errors in the current preview builds.
        # Standard FP32 execution on an RTX 5080 for a 9x9 game is extremely fast anyway.
        self.use_amp = False
        self.scaler  = GradScaler(enabled=False)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, valid_mask: np.ndarray,
                      greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection with illegal-move masking and tactical lookahead.

        Args:
            state      : np.array (2, 9, 9)
            valid_mask : np.bool array (81,) — True where move is legal
            greedy     : If True, uses tactical lookahead (Win/Block) then best Q

        Returns:
            action (int)
        """
        if not greedy and np.random.rand() < self.epsilon:
            valid_actions = np.where(valid_mask)[0]
            return int(np.random.choice(valid_actions))

        # 1. Proximity Algorithm & Center Prior (No hardcoding)
        # Gomoku requires local connectivity. We mask out actions far from the battle.
        occupied = state[0] + state[1]
        if np.sum(occupied) > 0:
            import torch.nn.functional as F
            occ_t = torch.FloatTensor(occupied).unsqueeze(0).unsqueeze(0).to(self.device)
            kernel = torch.ones((1, 1, 3, 3)).to(self.device) # Radius 1 (Strict Adjacency)
            prox_t = F.conv2d(occ_t, kernel, padding=1) > 0
            prox_mask = prox_t.cpu().numpy().flatten()
            
            restricted_mask = valid_mask & prox_mask
            if np.any(restricted_mask):
                valid_mask = restricted_mask

        # 2-Step Tactical Lookahead (Smart Filter)
        my_pieces  = state[0]
        opp_pieces = state[1]
        
        # 1. Can I win immediately?
        my_wins = self._find_winning_actions(my_pieces, opp_pieces)
        if my_wins: return my_wins[0]
        
        # 2. Must I block the opponent's immediate win?
        opp_wins = self._find_winning_actions(opp_pieces, my_pieces)
        if opp_wins: return opp_wins[0]
        
        # 3. Can I create an open-4 or fork? (Yields >= 2 winning spots next turn)
        if greedy:
            for action in np.where(valid_mask)[0]:
                r, c = divmod(action, self.board_size)
                my_pieces[r, c] = 1
                wins_next = self._find_winning_actions(my_pieces, opp_pieces)
                my_pieces[r, c] = 0
                if len(wins_next) >= 2:
                    return int(action)
                    
            # 4. Must I block opponent's open-4 or fork?
            for action in np.where(valid_mask)[0]:
                r, c = divmod(action, self.board_size)
                opp_pieces[r, c] = 1
                wins_next = self._find_winning_actions(opp_pieces, my_pieces)
                opp_pieces[r, c] = 0
                if len(wins_next) >= 2:
                    return int(action)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_opp_t = torch.FloatTensor(np.stack([state[1], state[0]])).unsqueeze(0).to(self.device)

        self.policy_net.eval()
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                q_my = self.policy_net(state_t).squeeze(0).float().cpu().numpy()
                q_opp = self.policy_net(state_opp_t).squeeze(0).float().cpu().numpy()
        self.policy_net.train()

        # Hybrid offense/defense: Value my attack + 1.5 * destroying his attack
        # (Makes the AI an extremely spiteful blocker)
        q_values = q_my + 1.5 * q_opp
        
        # Heavy Algorithmic Center-Gravity only for the very first opening moves
        empty_squares = np.sum(valid_mask)
        if empty_squares >= self.action_size - 2:
            for i in range(self.action_size):
                r, c = divmod(i, self.board_size)
                dist_to_center = abs(r - self.board_size//2) + abs(c - self.board_size//2)
                q_values[i] -= dist_to_center * 2.0  # Massive penalty to force center

        q_values[~valid_mask] = -1e9
        return int(np.argmax(q_values))

    def _find_winning_actions(self, b: np.ndarray, opp: np.ndarray) -> list[int]:
        """Returns all empty cells that would immediately complete a 5-in-a-row for board b."""
        n = self.board_size
        wins = []
        valid_mask = ((b + opp) == 0).flatten()
        for action in np.where(valid_mask)[0]:
            r, c = divmod(action, n)
            b[r, c] = 1
            if self._check_win(b):
                wins.append(int(action))
            b[r, c] = 0
        return wins

    def _check_win(self, b: np.ndarray) -> bool:
        """Fast 5-in-a-row check for a single 9x9 binary matrix."""
        n = self.board_size
        # Horizontal
        for r in range(n):
            for c in range(n - 4):
                if b[r, c:c+5].sum() == 5: return True
        # Vertical
        for r in range(n - 4):
            for c in range(n):
                if b[r:r+5, c].sum() == 5: return True
        # Diagonal \
        for r in range(n - 4):
            for c in range(n - 4):
                if np.trace(b[r:r+5, c:c+5]) == 5: return True
        # Diagonal /
        for r in range(n - 4):
            for c in range(4, n):
                if np.trace(np.fliplr(b[r:r+5, c-4:c+1])) == 5: return True
        return False

    # ------------------------------------------------------------------
    # Data augmentation (8-fold symmetry for 9x9 board)
    # ------------------------------------------------------------------

    def _augment_and_push(self, state, action, reward, next_state, done):
        """
        Store 8 symmetric copies of a transition (4 rotations × 2 flips).
        Each rotation/flip is applied identically to state, next_state, action.
        """
        n = self.board_size

        s  = state.copy()      # (2, 9, 9)
        ns = next_state.copy() # (2, 9, 9)

        row, col = divmod(action, n)

        for k in range(4):
            # Store original orientation
            act_k = (row * n + col)
            self.replay_buffer.push(s, act_k, reward, ns, done)

            # Store horizontal flip of this rotation
            s_flip  = np.flip(s,  axis=2).copy()
            ns_flip = np.flip(ns, axis=2).copy()
            col_flip = n - 1 - col
            act_flip = row * n + col_flip
            self.replay_buffer.push(s_flip, act_flip, reward, ns_flip, done)

            # Rotate 90° counter-clockwise for next iteration
            s  = np.rot90(s,  k=1, axes=(1, 2)).copy()
            ns = np.rot90(ns, k=1, axes=(1, 2)).copy()
            # Update (row, col) after rotation: (r,c) → (n-1-c, r)
            row, col = n - 1 - col, row

    def push(self, state, action, reward, next_state, done):
        """Store a transition with 8-fold augmentation."""
        self._augment_and_push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self) -> float:
        """
        Sample a mini-batch and perform one gradient update with AMP.

        Returns:
            loss (float), or 0.0 if buffer not ready.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        with autocast(enabled=self.use_amp):
            # Current Q
            q_current = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

            # Double DQN target
            with torch.no_grad():
                next_q_policy = self.policy_net(next_states_t)

                # Mask occupied cells
                occupied = (
                    next_states_t[:, 0, :, :] + next_states_t[:, 1, :, :]
                ).view(-1, self.action_size)
                next_q_policy[occupied > 0.5] = -1e9

                next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                q_next   = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
                q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

            loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Soft update target network
        self.train_step += 1
        for tgt, src in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)

        return loss.item()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Unwrap compiled model if needed
        policy_state = (
            self.policy_net._orig_mod.state_dict()
            if hasattr(self.policy_net, "_orig_mod")
            else self.policy_net.state_dict()
        )
        target_state = (
            self.target_net._orig_mod.state_dict()
            if hasattr(self.target_net, "_orig_mod")
            else self.target_net.state_dict()
        )
        torch.save({
            "policy_net": policy_state,
            "target_net": target_state,
            "optimizer":  self.optimizer.state_dict(),
            "scaler":     self.scaler.state_dict(),
            "epsilon":    self.epsilon,
            "train_step": self.train_step,
        }, path)
        print(f"[GomokuAgent] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        target = (
            self.policy_net._orig_mod
            if hasattr(self.policy_net, "_orig_mod")
            else self.policy_net
        )
        target.load_state_dict(ckpt["policy_net"])
        target_t = (
            self.target_net._orig_mod
            if hasattr(self.target_net, "_orig_mod")
            else self.target_net
        )
        target_t.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.epsilon    = ckpt.get("epsilon", self.epsilon_end)
        self.train_step = ckpt.get("train_step", 0)
        self.target_net.eval()
        print(f"[GomokuAgent] Loaded checkpoint ← {path}")

    # ------------------------------------------------------------------
    # Prediction API (coursework requirement)
    # ------------------------------------------------------------------

    def predict(self, board_state: np.ndarray):
        """
        Coursework API: given a raw board state, return the best move.

        Args:
            board_state : np.array of shape (9, 9) with values {0, 1, -1}
                          OR shape (2, 9, 9) already encoded.

        Returns:
            (row, col) : tuple of ints
        """
        if board_state.ndim == 2:
            state = np.zeros((2, 9, 9), dtype=np.float32)
            state[0] = (board_state == 1).astype(np.float32)
            state[1] = (board_state == -1).astype(np.float32)
            valid_mask = (board_state.flatten() == 0)
        else:
            state = board_state.astype(np.float32)
            occupied = (state[0] + state[1]).flatten()
            valid_mask = (occupied == 0)

        action = self.select_action(state, valid_mask, greedy=True)
        row, col = divmod(action, self.board_size)
        return int(row), int(col)
