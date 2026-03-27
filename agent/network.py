"""
Neural Network for 4x4 Connect-4 DQN Agent
============================================
Input : board state tensor of shape (batch, 2, 4, 4)
Output: Q-values for all 16 actions (row-major flattened board positions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectNet(nn.Module):
    """
    MLP Q-network for 4x4 Connect-4.

    For small grids (4x4), an MLP often outperforms small-kernel CNNs
    because it trivially captures global patterns (like full diagonals).
    """

    def __init__(self, board_size: int = 4, action_size: int = 16):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        input_size = 2 * board_size * board_size

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 2, 4, 4)
        Returns:
            q_values: Tensor of shape (batch, 16)
        """
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class DuelingConnectNet(nn.Module):
    """
    Dueling DQN MLP variant.
    """

    def __init__(self, board_size: int = 4, action_size: int = 16):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        input_size = 2 * board_size * board_size

        # Shared backbone
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

        # Value stream
        self.value_fc1 = nn.Linear(256, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # Advantage stream
        self.adv_fc1 = nn.Linear(256, 128)
        self.adv_fc2 = nn.Linear(128, action_size)

        self._init_weights()

    def _backbone(self, x):
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward(self, x):
        feat = self._backbone(x)

        value = F.relu(self.value_fc1(feat))
        value = self.value_fc2(value)  # (batch, 1)

        adv = F.relu(self.adv_fc1(feat))
        adv = self.adv_fc2(adv)  # (batch, action_size)

        # Combine: subtract mean advantage for identifiability
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
