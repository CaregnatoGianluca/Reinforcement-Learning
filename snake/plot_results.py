"""
Generate learning curve plots for the report.
Usage: python plot_results.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report', 'figures')


def smooth(data, window=100):
    """Moving average."""
    if len(data) < window:
        window = max(1, len(data))
    return np.convolve(data, np.ones(window) / window, mode='valid')


def load_log(algo, env_type):
    """Load training log, return rewards list or None."""
    path = os.path.join(WEIGHTS_DIR, f'{algo}_{env_type}_log.npz')
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)['rewards']


def plot_learning_curves(env_type, baseline_reward):
    """One subplot per algorithm, with baseline reference line."""
    algos = ['dqn', 'reinforce', 'actor_critic']
    names = {'dqn': 'DQN', 'reinforce': 'REINFORCE', 'actor_critic': 'Actor-Critic'}
    colors = {'dqn': '#2196F3', 'reinforce': '#FF5722', 'actor_critic': '#4CAF50'}

    logs = {a: load_log(a, env_type) for a in algos}
    logs = {a: r for a, r in logs.items() if r is not None}

    n = len(logs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (algo, rewards) in zip(axes, logs.items()):
        ax.plot(smooth(rewards), color=colors[algo], linewidth=2, label=names[algo])
        ax.plot(rewards, color=colors[algo], alpha=0.1, linewidth=0.5)
        ax.axhline(y=baseline_reward, color='#9C27B0', linestyle='--',
                   linewidth=2, label=f'Baseline ({baseline_reward:.4f})')
        ax.set_title(names[algo])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Avg Reward')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    env_label = 'Fully Observable' if 'fully' in env_type else 'Partially Observable'
    fig.suptitle(f'Learning Curves — {env_label}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'learning_curves_{env_type}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved learning_curves_{env_type}.png")


def plot_comparison(baseline_full, baseline_partial):
    """Side-by-side: fully vs partially observable for each algorithm."""
    algos = ['dqn', 'reinforce', 'actor_critic']
    names = {'dqn': 'DQN', 'reinforce': 'REINFORCE', 'actor_critic': 'Actor-Critic'}
    colors = {'dqn': '#2196F3', 'reinforce': '#FF5722', 'actor_critic': '#4CAF50'}

    fig, axes = plt.subplots(len(algos), 2, figsize=(12, 4 * len(algos)))

    for row, algo in enumerate(algos):
        for col, (env, bl) in enumerate([('fully_observable', baseline_full),
                                          ('partially_observable', baseline_partial)]):
            ax = axes[row][col]
            rewards = load_log(algo, env)
            if rewards is not None:
                ax.plot(smooth(rewards), color=colors[algo], linewidth=2, label=names[algo])
            ax.axhline(y=bl, color='#9C27B0', linestyle='--', linewidth=2, label=f'Baseline ({bl:.4f})')
            env_label = 'Fully Obs.' if 'fully' in env else 'Partially Obs.'
            ax.set_title(f'{names[algo]} — {env_label}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Avg Reward')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.suptitle('Fully vs Partially Observable Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'comparison_observability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved comparison_observability.png")


if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Baseline rewards (from evaluation)
    BASELINE_FULL = 0.1124
    BASELINE_PARTIAL = 0.0878

    print("Generating plots...")
    plot_learning_curves('fully_observable', BASELINE_FULL)
    plot_learning_curves('partially_observable', BASELINE_PARTIAL)
    plot_comparison(BASELINE_FULL, BASELINE_PARTIAL)
    print("Done!")
