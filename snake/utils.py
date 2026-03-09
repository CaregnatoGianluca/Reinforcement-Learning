"""
Utility functions for the Snake RL project.
Includes seed fixing, action masking, and episode logging.
"""
import random
import numpy as np
import tensorflow as tf


def set_all_seeds(seed=0):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_valid_actions_mask(boards):
    """
    For each board, compute which of the 4 actions are valid (won't hit a wall).
    Fully vectorized with numpy — no Python loops.

    Actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
    Returns: np.array of shape (n_boards, 4) with 1 for valid, 0 for invalid.
    """
    WALL = 0  # BaseEnvironment.WALL
    HEAD = 4  # BaseEnvironment.HEAD

    n_boards = boards.shape[0]
    board_size = boards.shape[1]
    mask = np.ones((n_boards, 4), dtype=np.float32)

    # Find head positions
    heads = np.argwhere(boards == HEAD) # [board_idx, row, col]

    # Extract row, col for each board's head
    head_rows = heads[:, 1]
    head_cols = heads[:, 2]
    board_indices = heads[:, 0]

    # Compute new positions for all 4 actions at once
    offsets_r = np.array([1, 0, -1, 0])   # row for UP, RIGHT, DOWN, LEFT
    offsets_c = np.array([0, 1, 0, -1])   # col for UP, RIGHT, DOWN, LEFT

    for action_idx in range(4):
        new_r = head_rows + offsets_r[action_idx]
        new_c = head_cols + offsets_c[action_idx]

        # Out of bounds check
        out_of_bounds = (new_r < 0) | (new_r >= board_size) | (new_c < 0) | (new_c >= board_size)
        mask[board_indices[out_of_bounds], action_idx] = 0.0

        # Wall check (only for in-bounds positions)
        in_bounds = ~out_of_bounds
        if np.any(in_bounds):
            is_wall = boards[board_indices[in_bounds], new_r[in_bounds], new_c[in_bounds]] == WALL
            mask[board_indices[in_bounds][is_wall], action_idx] = 0.0

    return mask


def get_valid_actions_mask_from_env(env):
    """Convenience: get action masks directly from an environment object."""
    return get_valid_actions_mask(env.boards)


class EpisodeLogger:
    """Track training metrics over episodes/iterations."""

    def __init__(self):
        self.rewards_history = []
        self.fruits_history = []
        self.wall_hits_history = []
        self.losses_history = []

    def log(self, rewards, fruits=None, wall_hits=None, loss=None):
        self.rewards_history.append(float(np.mean(rewards)))
        if fruits is not None:
            self.fruits_history.append(float(np.mean(fruits)))
        if wall_hits is not None:
            self.wall_hits_history.append(float(np.mean(wall_hits)))
        if loss is not None:
            self.losses_history.append(float(loss))

    def get_smoothed(self, data, window=100):
        """Smooth data with a moving average."""
        if len(data) < window:
            window = max(1, len(data))
        return np.convolve(data, np.ones(window) / window, mode='valid')

    def save(self, filepath):
        """Save logs to npz file."""
        np.savez(filepath,
                 rewards=self.rewards_history,
                 fruits=self.fruits_history,
                 wall_hits=self.wall_hits_history,
                 losses=self.losses_history)

    @classmethod
    def load(cls, filepath):
        """Load logs from npz file."""
        logger = cls()
        data = np.load(filepath, allow_pickle=True)
        logger.rewards_history = data['rewards'].tolist()
        logger.fruits_history = data['fruits'].tolist()
        logger.wall_hits_history = data['wall_hits'].tolist()
        logger.losses_history = data['losses'].tolist()
        return logger
