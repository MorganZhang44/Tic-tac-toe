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


class GomokuNet(nn.Module):
    """
    CNN Q-network designed for the 9x9 Gomoku board.
    It uses Conv2D layers to extract spatial 5-in-a-row features.
    """

    def __init__(self, board_size: int = 9, action_size: int = 81):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # 9x9 with padding=1 remains 9x9.
        conv_out_size = 128 * board_size * board_size

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_size)

        self._init_weights()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 2, 9, 9)
        Returns:
            q_values: Tensor of shape (batch, 81)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class DuelingGomokuNet(nn.Module):
    """
    Dueling DQN CNN variant for 9x9 Gomoku.
    """

    def __init__(self, board_size: int = 9, action_size: int = 81):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        # Shared Backbone (CNN)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        conv_out_size = 128 * board_size * board_size

        # Value stream
        self.value_fc1 = nn.Linear(conv_out_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Advantage stream
        self.adv_fc1 = nn.Linear(conv_out_size, 256)
        self.adv_fc2 = nn.Linear(256, action_size)

        self._init_weights()

    def _backbone(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1) # Flatten

    def forward(self, x):
        feat = self._backbone(x)

        value = F.relu(self.value_fc1(feat))
        value = self.value_fc2(value)  # (batch, 1)

        adv = F.relu(self.adv_fc1(feat))
        adv = self.adv_fc2(adv)  # (batch, action_size)

        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
