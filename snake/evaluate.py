"""
Evaluation script for Snake RL project.
Loads trained agent weights, runs evaluation, and compares with baseline.
TensorFlow/Keras implementation.

This is the CRITICAL submission file — must work with just:
    python evaluate.py
"""
import os
import sys
import numpy as np
import tensorflow as tf

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_all_seeds, get_valid_actions_mask_from_env
from environments_fully_observable import OriginalSnakeEnvironment as FullyObsEnv
from environments_partially_observable import OriginalSnakeEnvironment as PartiallyObsEnv
from baseline import run_baseline, greedy_bfs_action
from agents.dqn_agent import DQNAgent
from agents.reinforce_agent import REINFORCEAgent
from agents.actor_critic_agent import ActorCriticAgent


WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
N_BOARDS = 100
N_STEPS = 1000
BOARD_SIZE = 7
MASK_SIZE = 2


def evaluate_agent(agent, env, n_steps=1000, agent_name="Agent", greedy=True):
    """
    Evaluate a trained RL agent on an environment.

    Args:
        agent: trained agent with select_action_greedy method
        env: Snake environment
        n_steps: number of evaluation steps
        agent_name: name for display
        greedy: if True, use greedy actions (no exploration)

    Returns:
        dict with evaluation metrics
    """
    all_rewards = []
    fruits_eaten = 0
    wall_hits = 0
    self_hits = 0
    wins = 0

    for step in range(n_steps):
        # Get state (channels-last, no transpose needed)
        state = env.to_state()
        if hasattr(state, 'numpy'):
            state = state.numpy()

        # Get action mask
        mask = get_valid_actions_mask_from_env(env)

        # Select action (greedy, no exploration)
        if greedy:
            actions = agent.select_action_greedy(state, mask)
        else:
            actions = agent.select_action(state, mask)

        # Environment step
        rewards = env.move(actions.reshape(-1, 1))
        rewards_np = rewards.numpy().flatten()

        all_rewards.append(np.mean(rewards_np))
        fruits_eaten += np.sum(rewards_np == 0.5)
        wall_hits += np.sum(rewards_np == -0.1)
        self_hits += np.sum(rewards_np == -0.2)
        wins += np.sum(rewards_np == 1.0)

    results = {
        'avg_reward': np.mean(all_rewards),
        'total_reward': np.sum(all_rewards),
        'fruits_eaten': int(fruits_eaten),
        'wall_hits': int(wall_hits),
        'self_hits': int(self_hits),
        'wins': int(wins),
    }

    return results


def load_and_evaluate(algo, env_type, n_boards=N_BOARDS, n_steps=N_STEPS):
    """Load a trained agent and evaluate it."""
    # Create environment
    set_all_seeds(0)
    if env_type == 'fully_observable':
        env = FullyObsEnv(n_boards, BOARD_SIZE)
        C, H, W = 4, BOARD_SIZE, BOARD_SIZE
    else:
        env = PartiallyObsEnv(n_boards, BOARD_SIZE, MASK_SIZE)
        obs_size = 2 * MASK_SIZE + 1
        C, H, W = 4, obs_size, obs_size

    # Load agent
    if algo == 'dqn':
        agent = DQNAgent(C, H, W, 4)
        weight_path = os.path.join(WEIGHTS_DIR, f'dqn_{env_type}.h5')
    elif algo == 'reinforce':
        agent = REINFORCEAgent(C, H, W, 4)
        weight_path = os.path.join(WEIGHTS_DIR, f'reinforce_{env_type}.h5')
    elif algo == 'actor_critic':
        agent = ActorCriticAgent(C, H, W, 4)
        weight_path = os.path.join(WEIGHTS_DIR, f'actor_critic_{env_type}.h5')
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    if not os.path.exists(weight_path):
        print(f"  ⚠️  Weights not found: {weight_path}")
        return None

    agent.load(weight_path, eval_mode=True)

    # Evaluate
    results = evaluate_agent(agent, env, n_steps, f"{algo.upper()}")
    return results


def evaluate_baseline(env_type, n_boards=N_BOARDS, n_steps=N_STEPS):
    """Evaluate the BFS baseline heuristic."""
    set_all_seeds(0)
    if env_type == 'fully_observable':
        env = FullyObsEnv(n_boards, BOARD_SIZE)
        results = run_baseline(env, n_steps, verbose=False)
    else:
        env = PartiallyObsEnv(n_boards, BOARD_SIZE, MASK_SIZE)
        results = run_baseline(env, n_steps, verbose=False,
                               partially_observable=True, mask_size=MASK_SIZE)
    return results


def print_results_table(results_dict, title):
    """Print a formatted results table."""
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(f"  {'Agent':<25} {'Avg Reward':>12} {'Fruits':>8} {'Wall Hits':>10} {'Self Hits':>10} {'Wins':>6}")
    print(f"  {'-'*73}")
    for name, results in results_dict.items():
        if results is not None:
            self_hits = results.get('self_hits', 'N/A')
            wins = results.get('wins', 0)
            if isinstance(self_hits, (int, float)):
                print(f"  {name:<25} {results['avg_reward']:>12.4f} {results['fruits_eaten']:>8} {results['wall_hits']:>10} {self_hits:>10} {wins:>6}")
            else:
                print(f"  {name:<25} {results['avg_reward']:>12.4f} {results['fruits_eaten']:>8} {results['wall_hits']:>10} {'N/A':>10} {wins:>6}")
        else:
            print(f"  {name:<25} {'(no weights)':>12}")


def main():
    """Main evaluation function."""
    set_all_seeds(0)

    print("=" * 70)
    print("  SNAKE RL PROJECT — EVALUATION")
    print("=" * 70)
    print(f"  Boards: {N_BOARDS}, Steps: {N_STEPS}, Board size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"  Seed: 0")

    # ========== FULLY OBSERVABLE ==========
    print(f"\n\n{'#'*70}")
    print(f"  FULLY OBSERVABLE ENVIRONMENT")
    print(f"{'#'*70}")

    results_full = {}

    # Baseline
    print("\n  Evaluating Baseline (Greedy BFS)...")
    results_full['Baseline (BFS)'] = evaluate_baseline('fully_observable')

    # DQN
    print("  Evaluating DQN...")
    results_full['DQN'] = load_and_evaluate('dqn', 'fully_observable')

    # REINFORCE
    print("  Evaluating REINFORCE...")
    results_full['REINFORCE'] = load_and_evaluate('reinforce', 'fully_observable')

    # Actor-Critic (bonus)
    print("  Evaluating Actor-Critic...")
    results_full['Actor-Critic'] = load_and_evaluate('actor_critic', 'fully_observable')

    print_results_table(results_full, "FULLY OBSERVABLE — Results")

    # ========== PARTIALLY OBSERVABLE ==========
    print(f"\n\n{'#'*70}")
    print(f"  PARTIALLY OBSERVABLE ENVIRONMENT")
    print(f"{'#'*70}")

    results_partial = {}

    # Baseline
    print("\n  Evaluating Baseline (Greedy BFS)...")
    results_partial['Baseline (BFS)'] = evaluate_baseline('partially_observable')

    # DQN
    print("  Evaluating DQN...")
    results_partial['DQN'] = load_and_evaluate('dqn', 'partially_observable')

    # REINFORCE
    print("  Evaluating REINFORCE...")
    results_partial['REINFORCE'] = load_and_evaluate('reinforce', 'partially_observable')

    # Actor-Critic
    print("  Evaluating Actor-Critic...")
    results_partial['Actor-Critic'] = load_and_evaluate('actor_critic', 'partially_observable')

    print_results_table(results_partial, "PARTIALLY OBSERVABLE — Results")



if __name__ == '__main__':
    main()
