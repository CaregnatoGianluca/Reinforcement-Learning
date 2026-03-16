"""
Baseline Agent: Greedy BFS Heuristic (non-RL).
Finds shortest path from snake head to fruit using BFS,
avoiding walls and body segments. Falls back to a random safe action.
"""
import sys
import os
import numpy as np
from collections import deque
from tqdm import trange

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments_fully_observable import OriginalSnakeEnvironment as FullyObsEnv
from environments_partially_observable import OriginalSnakeEnvironment as PartiallyObsEnv
from utils import set_all_seeds


# Constants from the environment
HEAD = 4
BODY = 3
FRUIT = 2
EMPTY = 1
WALL = 0

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Action offsets: (row, col)
ACTION_OFFSETS = {
    UP:    (1, 0),
    RIGHT: (0, 1),
    DOWN:  (-1, 0),
    LEFT:  (0, -1),
}


def bfs_find_path(board, head_pos, fruit_pos):
    """
    BFS from head to fruit on a single board.
    Returns the first action to take, or None if no path exists.
    
    Args:
        board: 2D numpy array of shape (board_size, board_size)
        head_pos: tuple (row, col) of the head
        fruit_pos: tuple (row, col) of the fruit
    
    Returns:
        action (int) or None
    """
    board_size = board.shape[0]
    visited = set()
    visited.add(head_pos)
    
    # Queue entries: (row, col, first_action)
    queue = deque()
    
    # Seed the queue with all valid first moves
    for action, (dr, dc) in ACTION_OFFSETS.items():
        nr, nc = head_pos[0] + dr, head_pos[1] + dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            cell = board[nr, nc]
            if cell != WALL and cell != BODY:
                if (nr, nc) == fruit_pos:
                    return action  # Fruit is adjacent
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, action))
    
    # BFS
    while queue:
        r, c, first_action = queue.popleft()
        for action, (dr, dc) in ACTION_OFFSETS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < board_size and 0 <= nc < board_size and (nr, nc) not in visited:
                cell = board[nr, nc]
                if cell == WALL or cell == BODY:
                    continue
                if (nr, nc) == fruit_pos:
                    return first_action
                visited.add((nr, nc))
                queue.append((nr, nc, first_action))
    
    return None  # No path found


def get_safe_action(board, head_pos):
    """Get a random action that doesn't hit a wall or body."""
    safe_actions = []
    board_size = board.shape[0]
    for action, (dr, dc) in ACTION_OFFSETS.items():
        nr, nc = head_pos[0] + dr, head_pos[1] + dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            cell = board[nr, nc]
            if cell != WALL and cell != BODY:
                safe_actions.append(action)
    if safe_actions:
        return np.random.choice(safe_actions)
    # All actions are dangerous, pick any that doesn't hit a wall
    for action, (dr, dc) in ACTION_OFFSETS.items():
        nr, nc = head_pos[0] + dr, head_pos[1] + dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            if board[nr, nc] != WALL:
                return action
    # Completely stuck, just go UP
    return UP


def greedy_bfs_action(board):
    """
    Compute the best action for a single board using greedy BFS.
    Uses the full board for fully observable environment.
    
    Args:
        board: 2D numpy array of shape (board_size, board_size)
    
    Returns:
        action (int)
    """
    head_positions = np.argwhere(board == HEAD)
    fruit_positions = np.argwhere(board == FRUIT)
    
    if len(head_positions) == 0 or len(fruit_positions) == 0:
        return UP  # Fallback
    
    head_pos = tuple(head_positions[0])
    fruit_pos = tuple(fruit_positions[0])
    
    # Try BFS to fruit
    action = bfs_find_path(board, head_pos, fruit_pos)
    if action is not None:
        return action
    
    # No path to fruit, pick a safe action
    return get_safe_action(board, head_pos)


def get_local_view(board, head_pos, mask_size=2):
    """
    Extract the local view around the head, same as the partially
    observable environment's to_state() method.
    
    Returns a (2*mask_size+1, 2*mask_size+1) array.
    Areas outside the board are filled with WALL.
    """
    board_size = board.shape[0]
    view_size = 2 * mask_size + 1
    local = np.zeros((view_size, view_size))
    
    hr, hc = head_pos
    for dr in range(-mask_size, mask_size + 1):
        for dc in range(-mask_size, mask_size + 1):
            r, c = hr + dr, hc + dc
            vr, vc = dr + mask_size, dc + mask_size
            if 0 <= r < board_size and 0 <= c < board_size:
                local[vr, vc] = board[r, c]
            else:
                local[vr, vc] = WALL  # Out of bounds = wall
    
    return local


def greedy_bfs_action_partial(board, mask_size=2):
    """
    Compute the best action for a single board using ONLY the local view.
    Fair baseline for partially observable environment.
    
    - If the fruit is visible in the local window: BFS to it
    - If the fruit is NOT visible: take a random safe action
    
    Args:
        board: 2D numpy array (full board — we extract local view ourselves)
        mask_size: radius of the local view (default 2 → 5x5 window)
    
    Returns:
        action (int)
    """
    head_positions = np.argwhere(board == HEAD)
    if len(head_positions) == 0:
        return UP
    
    head_pos = tuple(head_positions[0])
    
    # Extract local view
    local_view = get_local_view(board, head_pos, mask_size)
    
    # Head is at center of local view
    local_head = (mask_size, mask_size)
    
    # Check if fruit is visible in local view
    fruit_positions = np.argwhere(local_view == FRUIT)
    
    if len(fruit_positions) > 0:
        # Fruit is visible, BFS on local view
        local_fruit = tuple(fruit_positions[0])
        action = bfs_find_path(local_view, local_head, local_fruit)
        if action is not None:
            return action
    
    # Fruit not visible or no path, take a random safe action on local view
    return get_safe_action(local_view, local_head)


def run_baseline(env, n_steps=1000, verbose=True, partially_observable=False, mask_size=2):
    """
    Run the greedy BFS baseline on an environment.
    
    Args:
        env: Snake environment instance
        n_steps: number of steps to simulate
        verbose: whether to show progress bar
    
    Returns:
        dict with 'rewards', 'total_reward', 'avg_reward', 'fruits_eaten'
    """
    all_rewards = []
    fruits_eaten = 0
    wall_hits = 0
    self_hits = 0
    wins = 0
    
    iterator = trange(n_steps, desc="Baseline (BFS)") if verbose else range(n_steps)
    
    for step in iterator:
        # Compute BFS action for each board
        actions = []
        for board in env.boards:
            if partially_observable:
                actions.append(greedy_bfs_action_partial(board, mask_size))
            else:
                actions.append(greedy_bfs_action(board))
        actions = np.array(actions).reshape(-1, 1)
        
        rewards = env.move(actions)
        rewards_np = rewards.numpy().flatten()
        all_rewards.append(np.mean(rewards_np))
        
        # Count fruits, wall hits, and self hits
        fruits_eaten += np.sum(rewards_np == 0.5)
        wall_hits += np.sum(rewards_np == -0.1)
        self_hits += np.sum(rewards_np == -0.2)
        wins += np.sum(rewards_np == 1.0)
        
        if verbose and step % 100 == 0:
            iterator.set_postfix({
                'avg_reward': f'{np.mean(all_rewards[-100:]):.4f}',
                'fruits': fruits_eaten,
                'wall_hits': wall_hits
            })
    
    results = {
        'rewards': all_rewards,
        'total_reward': sum(all_rewards),
        'avg_reward': np.mean(all_rewards),
        'fruits_eaten': int(fruits_eaten),
        'wall_hits': int(wall_hits),
        'self_hits': int(self_hits),
        'wins': int(wins),
    }
    
    return results


def main():
    """Run the baseline on both fully and partially observable environments."""
    set_all_seeds(0)
    
    print("=" * 60)
    print("BASELINE EVALUATION: Greedy BFS Heuristic")
    print("=" * 60)
    
    n_boards = 100
    n_steps = 1000
    board_size = 7
    
    # Fully Observable
    print(f"\n{'='*60}")
    print(f"Fully Observable Environment ({n_boards} boards, {n_steps} steps)")
    print(f"{'='*60}")
    env_full = FullyObsEnv(n_boards, board_size)
    results_full = run_baseline(env_full, n_steps)
    print(f"\n  Average Reward:  {results_full['avg_reward']:.4f}")
    print(f"  Fruits Eaten:    {results_full['fruits_eaten']}")
    print(f"  Wall Hits:       {results_full['wall_hits']}")
    
    # Partially Observable
    print(f"\n{'='*60}")
    print(f"Partially Observable Environment ({n_boards} boards, {n_steps} steps)")
    print(f"{'='*60}")
    set_all_seeds(0)
    env_partial = PartiallyObsEnv(n_boards, board_size, 2)
    # Fair baseline: only uses the local 5×5 view, not the full board
    results_partial = run_baseline(env_partial, n_steps, partially_observable=True, mask_size=2)
    print(f"\n  Average Reward:  {results_partial['avg_reward']:.4f}")
    print(f"  Fruits Eaten:    {results_partial['fruits_eaten']}")
    print(f"  Wall Hits:       {results_partial['wall_hits']}")
    
    return results_full, results_partial


if __name__ == "__main__":
    main()
