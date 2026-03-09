"""
Training script for Snake RL agents.

python train.py --algo dqn --env fully_observable --iterations 100000
python train.py --algo reinforce --env fully_observable --n_boards 200
python train.py --algo actor_critic --env fully_observable --n_boards 200

python train.py --algo dqn --env partially_observable --iterations 100000
python train.py --algo reinforce --env partially_observable --n_boards 200
python train.py --algo actor_critic --env partially_observable --n_boards 200
"""
import os
import sys
import argparse
import numpy as np
import random
import tensorflow as tf
from tqdm import trange


from environments_fully_observable import OriginalSnakeEnvironment as FullyObsEnv
from environments_partially_observable import OriginalSnakeEnvironment as PartiallyObsEnv
from agents.dqn_agent import DQNAgent
from agents.reinforce_agent import REINFORCEAgent
from agents.actor_critic_agent import ActorCriticAgent
from utils import set_all_seeds, get_valid_actions_mask_from_env, EpisodeLogger


def get_env(env_type, n_boards, board_size=7, mask_size=2):
    """Create the appropriate environment."""
    if env_type == 'fully_observable':
        return FullyObsEnv(n_boards, board_size)
    elif env_type == 'partially_observable':
        return PartiallyObsEnv(n_boards, board_size, mask_size)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def get_state_shape(env_type, board_size=7, mask_size=2):
    """Get the state dimensions for network initialization."""
    if env_type == 'fully_observable':
        return 4, board_size, board_size  # Channels, H, W
    else:
        obs_size = 2 * mask_size + 1
        return 4, obs_size, obs_size


def train_dqn(args):
    """Train DQN agent."""
    print(f"\n{'='*60}")
    print(f"Training DQN on {args.env} environment")
    print(f"{'='*60}\n")

    C, H, W = get_state_shape(args.env, args.board_size, args.mask_size)
    env = get_env(args.env, args.n_boards, args.board_size, args.mask_size)

    agent = DQNAgent(
        input_channels=C, board_h=H, board_w=W, n_actions=4,
        lr=args.lr, gamma=args.gamma,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=args.epsilon_decay,
        buffer_size=500000, batch_size=args.batch_size,
        target_update_freq=args.target_update_freq, tau=0.005,
        double_dqn=True, use_action_mask=True
    )

    logger = EpisodeLogger()
    best_avg_reward = -float('inf')

    # Training loop
    updates_per_step = 4  # Number of gradient updates per environment step

    for iteration in trange(args.iterations, desc="DQN Training"):
        # Get state
        state = env.to_state()
        if hasattr(state, 'numpy'):
            state = state.numpy()

        # Get action mask
        mask = get_valid_actions_mask_from_env(env)

        # Select action (ε-greedy)
        actions = agent.select_action(state, mask)

        # Environment step
        rewards = env.move(actions.reshape(-1, 1))
        rewards_np = rewards.numpy().flatten()

        # Get next state and mask
        next_state = env.to_state()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()
        next_mask = get_valid_actions_mask_from_env(env)

        # Store transitions in replay buffer
        agent.store_transition(state, actions, rewards_np, next_state, mask, next_mask)

        # Train (multiple updates per step for efficiency)
        loss = None
        for _ in range(updates_per_step):
            loss = agent.train_step_fn()

        # Logging
        avg_reward = np.mean(rewards_np)
        logger.log(rewards_np, loss=loss)

        # Save best model
        if iteration > 100 and iteration % 100 == 0:
            recent_avg = np.mean(logger.rewards_history[-100:])
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                weight_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'weights', f'dqn_{args.env}.h5'
                )
                agent.save(weight_path)

        if iteration % 500 == 0:
            recent = np.mean(logger.rewards_history[-100:]) if len(logger.rewards_history) >= 100 else np.mean(logger.rewards_history)
            print(f"\n  Iter {iteration}: avg_reward={recent:.4f}, epsilon={agent.epsilon:.3f}, buffer={len(agent.replay_buffer)}")

    # Save training logs
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'weights', f'dqn_{args.env}_log.npz'
    )
    logger.save(log_path)

    print(f"\nTraining complete! Best avg reward: {best_avg_reward:.4f}")
    return agent, logger


def train_reinforce(args):
    """Train REINFORCE with baseline agent."""
    print(f"\n{'='*60}")
    print(f"Training REINFORCE on {args.env} environment")
    print(f"{'='*60}\n")

    C, H, W = get_state_shape(args.env, args.board_size, args.mask_size)
    env = get_env(args.env, args.n_boards, args.board_size, args.mask_size)

    agent = REINFORCEAgent(
        input_channels=C, board_h=H, board_w=W, n_actions=4,
        lr_policy=args.lr, lr_value=args.lr * 3,
        gamma=args.gamma, rollout_length=args.rollout_length,
        use_whitening=True, use_action_mask=True,
        entropy_coef=0.01
    )

    get_mask_fn = get_valid_actions_mask_from_env

    logger = EpisodeLogger()
    best_avg_reward = -float('inf')

    for iteration in trange(args.iterations, desc="REINFORCE Training"):
        result = agent.train_rollout(env, get_mask_fn)

        logger.log([result['avg_reward']], loss=result['policy_loss'])

        # Save best model
        if iteration > 10 and iteration % 10 == 0:
            recent_avg = np.mean(logger.rewards_history[-10:])
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                weight_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'weights', f'reinforce_{args.env}.h5'
                )
                agent.save(weight_path)

        if iteration % 100 == 0:
            recent = np.mean(logger.rewards_history[-10:]) if len(logger.rewards_history) >= 10 else np.mean(logger.rewards_history)
            print(f"\n  Iter {iteration}: avg_reward={recent:.4f}, policy_loss={result['policy_loss']:.4f}, entropy={result['entropy']:.4f}")

    # Save training logs
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'weights', f'reinforce_{args.env}_log.npz'
    )
    logger.save(log_path)

    print(f"\nTraining complete! Best avg reward: {best_avg_reward:.4f}")
    return agent, logger


def train_actor_critic(args):
    """Train Actor-Critic agent."""
    print(f"\n{'='*60}")
    print(f"Training Actor-Critic on {args.env} environment")
    print(f"{'='*60}\n")

    C, H, W = get_state_shape(args.env, args.board_size, args.mask_size)
    env = get_env(args.env, args.n_boards, args.board_size, args.mask_size)

    agent = ActorCriticAgent(
        input_channels=C, board_h=H, board_w=W, n_actions=4,
        lr=args.lr, gamma=args.gamma, use_action_mask=True,
        entropy_coef=0.01, n_steps=args.rollout_length
    )

    get_mask_fn = get_valid_actions_mask_from_env

    logger = EpisodeLogger()
    best_avg_reward = -float('inf')

    for iteration in trange(args.iterations, desc="Actor-Critic Training"):
        result = agent.train_n_steps(env, get_mask_fn)

        logger.log([result['avg_reward']], loss=result['policy_loss'])

        if iteration > 10 and iteration % 10 == 0:
            recent_avg = np.mean(logger.rewards_history[-10:])
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                weight_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'weights', f'actor_critic_{args.env}.h5'
                )
                agent.save(weight_path)

        if iteration % 100 == 0:
            recent = np.mean(logger.rewards_history[-10:]) if len(logger.rewards_history) >= 10 else np.mean(logger.rewards_history)
            print(f"\n  Iter {iteration}: avg_reward={recent:.4f}, policy_loss={result['policy_loss']:.4f}, entropy={result['entropy']:.4f}")

    # Save training logs
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'weights', f'actor_critic_{args.env}_log.npz'
    )
    logger.save(log_path)

    print(f"\nTraining complete! Best avg reward: {best_avg_reward:.4f}")
    return agent, logger


def main():
    parser = argparse.ArgumentParser(description='Train Snake RL agents')
    parser.add_argument('--algo', type=str, default='dqn',choices=['dqn', 'reinforce', 'actor_critic'],help='Algorithm to train')
    parser.add_argument('--env', type=str, default='fully_observable',choices=['fully_observable', 'partially_observable'],help='Environment type')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--n_boards', type=int, default=1000, help='Number of parallel boards')
    parser.add_argument('--board_size', type=int, default=7, help='Board size (including walls)')
    parser.add_argument('--mask_size', type=int, default=2, help='Mask size for partially observable env')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for DQN')
    parser.add_argument('--epsilon_decay', type=int, default=2000, help='Epsilon decay steps for DQN')
    parser.add_argument('--target_update_freq', type=int, default=200, help='Target network update frequency for DQN')
    parser.add_argument('--rollout_length', type=int, default=50, help='Rollout length for REINFORCE/Actor-Critic')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    # Fix seeds
    set_all_seeds(args.seed)

    # Create weights directory
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    print(f"Configuration:")
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env}")
    print(f"Iterations: {args.iterations}")
    print(f"N boards: {args.n_boards}")
    print(f"Board size: {args.board_size}")
    print(f"Seed: {args.seed}")
    print(f"LR: {args.lr}")
    print(f"Gamma: {args.gamma}")

    if args.algo == 'dqn':
        train_dqn(args)
    elif args.algo == 'reinforce':
        train_reinforce(args)
    elif args.algo == 'actor_critic':
        train_actor_critic(args)


if __name__ == '__main__':
    main()
