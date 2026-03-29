import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingGomokuNet(nn.Module):
    """
    Standard CNN architecture for 9x9 Gomoku (matching the 85MB "best" weights).
    Architecture: 3 Conv layers + Dueling MLP Heads.
    """
    def __init__(self, board_size=9, action_size=81):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        # Shared CNN Backbone (Matches 85MB weights)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Dueling Heads (Value & Advantage)
        # Note: 128 channels * 9 * 9 = 10368
        self.value_fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self.adv_fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.adv_fc2 = nn.Linear(256, action_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _backbone(self, x):
        """
        Args:
            x: Tensor of shape (batch, 2, 9, 9)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1) # Flatten

    def forward(self, x):
        feat = self._backbone(x)
        
        # Value Stream
        v_h = F.relu(self.value_fc1(feat))
        value = self.value_fc2(v_h) # (batch, 1)
        
        # Adv Stream
        a_h = F.relu(self.adv_fc1(feat))
        adv = self.adv_fc2(a_h) # (batch, action_size)
        
        # Combine: Q = V + A - mean(A)
        return value + adv - adv.mean(dim=1, keepdim=True)
