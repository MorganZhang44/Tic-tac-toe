"""
CNN Network for 9x9 Gomoku DQN Agent
======================================
Input : board state tensor of shape (batch, 2, 9, 9)
Output: Q-values for all 81 actions (row-major flattened board positions)

Architecture: Convolutional backbone + Residual blocks + Dueling heads.
Uses CNN because 9x9 grids require capturing local spatial patterns
(3/4/5-in-a-row threats): CNNs get this for free via weight sharing,
whereas MLP would need to learn each position independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard residual block: Conv → BN → ReLU → Conv → BN → skip.
    Preserves spatial dimensions via same-padding.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DuelingGomokuNet(nn.Module):
    """
    Dueling CNN network for 9x9 Gomoku.

    Backbone: 1 initial conv layer + N residual blocks (default 6).
    Heads:
      - Value:     GlobalAvgPool → FC(128) → FC(1)
      - Advantage: conv(2 ch) → flatten → FC(81)

    Final Q = V + A - mean(A)  (standard dueling combination)
    """

    def __init__(
        self,
        board_size: int = 9,
        action_size: int = 81,
        channels: int = 128,
        num_res_blocks: int = 6,
    ):
        super().__init__()
        self.board_size  = board_size
        self.action_size = action_size

        # ── Initial convolution ────────────────────────────────────────
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ── Residual tower ────────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # ── Value head ────────────────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * board_size * board_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # ── Advantage head ────────────────────────────────────────────
        self.advantage_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, action_size),
        )

        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 2, 9, 9)
        Returns:
            q_values: Tensor of shape (batch, 81)
        """
        feat = self.input_conv(x)
        feat = self.res_blocks(feat)

        value = self.value_head(feat)                           # (B, 1)
        advantage = self.advantage_head(feat)                   # (B, 81)

        # Dueling combination
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
