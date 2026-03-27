"""
Experience Replay Buffer
========================
Stores (state, action, reward, next_state, done) transitions and
supports uniform random sampling for DQN training.
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Fixed-size FIFO experience replay buffer.

    Args:
        capacity : Maximum number of transitions to store.
        seed     : Optional random seed for reproducibility.
    """

    def __init__(self, capacity: int = 50_000, seed: int = 42):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    def push(self, state, action: int, reward: float,
             next_state, done: bool):
        """Add a single transition to the buffer."""
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        """
        Uniformly sample a mini-batch.

        Returns:
            states      : np.array (B, 2, 4, 4)
            actions     : np.array (B,)
            rewards     : np.array (B,)
            next_states : np.array (B, 2, 4, 4)
            dones       : np.array (B,)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size


class PrioritisedReplayBuffer:
    """
    Prioritised Experience Replay (PER) buffer.
    Samples transitions proportional to their TD error.

    Args:
        capacity : Maximum buffer size.
        alpha    : Priority exponent (0 = uniform, 1 = full priority).
        beta     : Importance sampling exponent (annealed to 1 during training).
        seed     : Random seed.
    """

    def __init__(self, capacity: int = 50_000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 seed: int = 42):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        random.seed(seed)
        np.random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = None):
        if beta is None:
            beta = self.beta
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size,
                                   replace=False, p=probs)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Importance-sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(float(err)) + 1e-6

    def __len__(self):
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size
