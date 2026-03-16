# Evaluation script for the Snake RL project
# Run with: python evaluate.py
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_all_seeds, get_valid_actions_mask_from_env
from environments_fully_observable import OriginalSnakeEnvironment as FullyObsEnv
from environments_partially_observable import OriginalSnakeEnvironment as PartiallyObsEnv
from baseline import run_baseline
from agents.dqn_agent import DQNAgent


WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')


def evaluate_agent(agent, env, n_steps):
    """Run the agent on the env for n_steps and return metrics."""
    all_rewards = []
    fruits_eaten = 0
    wall_hits = 0
    self_hits = 0
    wins = 0

    for _ in range(n_steps):
        state = env.to_state()
        if hasattr(state, 'numpy'):
            state = state.numpy()
        mask = get_valid_actions_mask_from_env(env)
        actions = agent.select_action_greedy(state, mask)

        rewards = env.move(actions.reshape(-1, 1)).numpy().flatten()

        all_rewards.append(np.mean(rewards))
        fruits_eaten += np.sum(rewards == 0.5)
        wall_hits += np.sum(rewards == -0.1)
        self_hits += np.sum(rewards == -0.2)
        wins += np.sum(rewards == 1.0)

    return {
        'avg_reward': np.mean(all_rewards),
        'fruits_eaten': int(fruits_eaten),
        'wall_hits': int(wall_hits),
        'self_hits': int(self_hits),
        'wins': int(wins),
    }


def load_and_evaluate(algo, env_type):
    """Load a trained agent and evaluate it on 100 boards for 1000 steps."""
    set_all_seeds(0)

    if env_type == 'fully_observable':
        env = FullyObsEnv(100, 7)
        C, H, W = 4, 7, 7
    else:
        env = PartiallyObsEnv(100, 7, 2)
        C, H, W = 4, 5, 5

    # create the right agent and find the weights file
    if algo == 'dqn':
        agent = DQNAgent(C, H, W, 4)

    weight_path = os.path.join(WEIGHTS_DIR, f'{algo}_{env_type}.h5')
    if not os.path.exists(weight_path):
        print(f"WARNING: weights not found at {weight_path}")
        return None

    agent.load(weight_path, eval_mode=True)
    return evaluate_agent(agent, env, 1000)


def print_results(name, res):
    """Print results for one agent."""
    if res is None:
        print(f"  {name}: no weights found\n")
        return
    print(f"  {name}: avg_reward={res['avg_reward']:.4f}, "
          f"fruits={res['fruits_eaten']}, wall_hits={res['wall_hits']}, "
          f"self_hits={res['self_hits']}, wins={res['wins']}\n")


def main():
    set_all_seeds(0)

    algos = ['dqn']

    for env_type in ['fully_observable', 'partially_observable']:
        print(f"\n{'='*60}")
        print(f"  {env_type.upper().replace('_', ' ')}")
        print(f"{'='*60}")

        # baseline
        set_all_seeds(0)
        if env_type == 'fully_observable':
            env = FullyObsEnv(100, 7)
            bl = run_baseline(env, 1000, verbose=False)
        else:
            env = PartiallyObsEnv(100, 7, 2)
            bl = run_baseline(env, 1000, verbose=False, partially_observable=True, mask_size=2)
        print_results('Baseline (BFS)', bl)

        # rl agents
        for algo in algos:
            res = load_and_evaluate(algo, env_type)
            print_results(algo.upper(), res)


if __name__ == '__main__':
    main()
