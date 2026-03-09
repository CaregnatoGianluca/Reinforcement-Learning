"""
REINFORCE with Baseline Agent for Snake.
Monte Carlo policy gradient with a learned value-function baseline for
variance reduction. Adapted for continuous (non-episodic) environments
using bootstrapped returns at the end of each rollout.
"""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.networks import PolicyNetwork, ValueNetwork


class REINFORCEAgent:
    """
    REINFORCE with Baseline (Monte Carlo Policy Gradient).

    Since the Snake environment is continuous (no episode termination),
    we collect fixed-length rollouts and bootstrap V(s_T) at the end
    to estimate future returns beyond the rollout window.
    """

    def __init__(self, input_channels=4, board_h=7, board_w=7, n_actions=4,
                 lr_policy=5e-4, lr_value=1.5e-3, gamma=0.95,
                 rollout_length=50, use_whitening=True,
                 use_action_mask=True, entropy_coef=0.01):

        self.n_actions = n_actions
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.use_whitening = use_whitening
        self.use_action_mask = use_action_mask
        self.entropy_coef = entropy_coef

        # Policy network: π(a|s; θ)
        self.policy = PolicyNetwork(input_channels, board_h, board_w, n_actions)
        # Value network (baseline): V(s; w)
        self.value = ValueNetwork(input_channels, board_h, board_w)

        # Build networks with dummy input (channels-last)
        dummy = tf.zeros((1, board_h, board_w, input_channels))
        self.policy(dummy)
        self.value(dummy)

        # Separate optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_policy)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_value)

    def select_action(self, state, action_mask=None):
        """
        Sample action from policy π(a|s; θ).

        Args:
            state: numpy array (n_boards, H, W, C) — channels-last
            action_mask: numpy array (n_boards, 4) with 1=valid, 0=invalid

        Returns:
            actions: numpy array (n_boards,)
        """
        state_t = tf.constant(state, dtype=tf.float32)
        mask_t = None
        if action_mask is not None and self.use_action_mask:
            mask_t = tf.constant(action_mask, dtype=tf.float32)

        probs = self.policy(state_t, action_mask=mask_t)
        # Sample from categorical distribution
        logits = tf.math.log(probs + 1e-8)
        actions = tf.random.categorical(logits, 1)  # (N, 1)
        actions = tf.squeeze(actions, axis=-1)  # (N,)

        return actions.numpy()

    def compute_returns(self, rewards, bootstrap_value=None):
        """
        Compute discounted returns G_t for each timestep.

        For continuous environments (no episode termination), we bootstrap
        the value at the end: G_T = V(s_T), then work backwards:
        G_t = r_t + γ * G_{t+1}

        Args:
            rewards: list of numpy arrays, each (n_boards,)
            bootstrap_value: numpy array (n_boards,) — V(s_T) estimate

        Returns:
            returns: numpy array (rollout_length, n_boards)
        """
        T = len(rewards)
        n_boards = rewards[0].shape[0]
        returns = np.zeros((T, n_boards), dtype=np.float32)

        # Bootstrap: instead of G=0, use V(s_T) to account for
        # future rewards beyond the rollout window
        if bootstrap_value is not None:
            G = bootstrap_value.copy()
        else:
            G = np.zeros(n_boards, dtype=np.float32)

        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        return returns

    def train_rollout(self, env, get_mask_fn=None):
        """
        Collect a fixed-length rollout and perform REINFORCE update.
        Uses bootstrapped V(s_T) to handle the non-episodic environment.

        Returns:
            dict with 'policy_loss', 'avg_reward', 'value_loss', 'entropy'
        """
        saved_states = []
        saved_actions = []
        saved_masks = []
        saved_rewards = []
        total_rewards = []

        for t in range(self.rollout_length):
            # Get current state (channels-last, no transpose needed)
            state = env.to_state()
            if hasattr(state, 'numpy'):
                state = state.numpy()

            # Get action mask
            mask = None
            if get_mask_fn is not None and self.use_action_mask:
                mask = get_mask_fn(env)

            # Select action from policy
            actions = self.select_action(state, mask)

            # Environment step
            rewards = env.move(actions.reshape(-1, 1))
            rewards_np = rewards.numpy().flatten()

            # Store
            saved_states.append(state)
            saved_actions.append(actions)
            saved_masks.append(mask)
            saved_rewards.append(rewards_np)
            total_rewards.append(np.mean(rewards_np))

        # Bootstrap: estimate V(s_T) for the state AFTER the rollout
        last_state = env.to_state()
        if hasattr(last_state, 'numpy'):
            last_state = last_state.numpy()
        last_state_t = tf.constant(last_state, dtype=tf.float32)
        bootstrap_value = tf.squeeze(self.value(last_state_t), axis=-1).numpy()

        # Compute returns with bootstrap
        returns = self.compute_returns(saved_rewards, bootstrap_value)  # (T, n_boards)

        # Normalize returns
        if self.use_whitening:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        returns_t = tf.constant(returns, dtype=tf.float32)

        # Stack all states: (T, N, H, W, C)
        all_states = np.array(saved_states)
        T, N = all_states.shape[0], all_states.shape[1]
        all_states_flat = all_states.reshape(T * N, *all_states.shape[2:])
        all_states_flat_t = tf.constant(all_states_flat, dtype=tf.float32)

        # Value loss (baseline update)
        with tf.GradientTape() as value_tape:
            values = tf.squeeze(self.value(all_states_flat_t), axis=-1)  # (T*N,)
            values = tf.reshape(values, (T, N))  # (T, N)
            value_loss = tf.reduce_mean(tf.square(values - tf.stop_gradient(returns_t)))

        value_grads = value_tape.gradient(value_loss, self.value.trainable_variables)
        value_grads, _ = tf.clip_by_global_norm(value_grads, 5.0)
        self.value_optimizer.apply_gradients(zip(value_grads, self.value.trainable_variables))

        # Policy loss (REINFORCE with baseline)
        with tf.GradientTape() as policy_tape:
            # Recompute values (detached) for advantage
            values_detached = tf.squeeze(self.value(all_states_flat_t), axis=-1)
            values_detached = tf.stop_gradient(tf.reshape(values_detached, (T, N)))
            advantages = returns_t - values_detached

            # Compute log probs for taken actions
            policy_loss = tf.constant(0.0)
            entropy_sum = tf.constant(0.0)

            for t in range(T):
                state_t = tf.constant(saved_states[t], dtype=tf.float32)
                mask_t = None
                if saved_masks[t] is not None:
                    mask_t = tf.constant(saved_masks[t], dtype=tf.float32)

                logits = self.policy.get_logits(state_t, action_mask=mask_t)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                probs = tf.nn.softmax(logits, axis=-1)

                # Gather log probs for chosen actions
                actions_t = tf.constant(saved_actions[t], dtype=tf.int64)
                indices = tf.stack([tf.range(N, dtype=tf.int64), actions_t], axis=1)
                log_prob_selected = tf.gather_nd(log_probs, indices)

                # Policy gradient loss: -log π(a|s) * A
                policy_loss -= tf.reduce_mean(log_prob_selected * advantages[t])

                # Entropy
                entropy_sum += tf.reduce_mean(-tf.reduce_sum(probs * log_probs, axis=-1))

            policy_loss = policy_loss / T
            entropy = entropy_sum / T
            total_policy_loss = policy_loss - self.entropy_coef * entropy

        policy_grads = policy_tape.gradient(total_policy_loss, self.policy.trainable_variables)
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, 5.0)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        return {
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'avg_reward': np.mean(total_rewards),
            'entropy': entropy.numpy(),
        }

    def select_action_greedy(self, state, action_mask=None):
        """Select the most probable action (no sampling) — for evaluation."""
        state_t = tf.constant(state, dtype=tf.float32)
        mask_t = None
        if action_mask is not None and self.use_action_mask:
            mask_t = tf.constant(action_mask, dtype=tf.float32)

        probs = self.policy(state_t, action_mask=mask_t)
        return tf.argmax(probs, axis=-1).numpy()

    def save(self, filepath):
        """Save policy network weights."""
        self.policy.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath, eval_mode=True):
        """Load policy network weights."""
        self.policy.load_weights(filepath)
        print(f"Model loaded from {filepath}")
