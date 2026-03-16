"""
DQN Agent for Snake — Deep Q-Network with Experience Replay and Target Network.
Uses Double DQN, ε-greedy exploration, and Polyak-averaged target network.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.networks import QNetwork


class ReplayBuffer:
    """
    Experience Replay Buffer.
    Stores (s, a, r, s') transitions and samples random mini-batches.
    This breaks temporal correlations and makes training data approximately IID.
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, mask=None, next_mask=None):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, mask, next_mask))

    def push_batch(self, states, actions, rewards, next_states, masks=None, next_masks=None):
        """Store a batch of transitions (from parallel environments)."""
        batch_size = states.shape[0]
        for i in range(batch_size):
            m = masks[i] if masks is not None else None
            nm = next_masks[i] if next_masks is not None else None
            self.buffer.append((
                states[i], actions[i], rewards[i],
                next_states[i], m, nm
            ))

    def sample(self, batch_size):
        """Sample a random mini-batch."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])

        has_masks = batch[0][4] is not None
        if has_masks:
            masks = np.array([t[4] for t in batch])
            next_masks = np.array([t[5] for t in batch])
        else:
            masks = None
            next_masks = None

        return states, actions, rewards, next_states, masks, next_masks

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with experience replay, target network, and Double DQN."""

    def __init__(self, input_channels=4, board_h=7, board_w=7, n_actions=4,
                 lr=1e-4, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=2000, buffer_size=500000, batch_size=256,
                 target_update_freq=200, tau=0.005, double_dqn=True,
                 use_action_mask=True):

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.double_dqn = double_dqn
        self.use_action_mask = use_action_mask

        # Epsilon-greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Q-Network and Target Network
        self.q_network = QNetwork(input_channels, board_h, board_w, n_actions)
        self.target_network = QNetwork(input_channels, board_h, board_w, n_actions)

        # Build networks by calling with dummy input (channels-last)
        dummy = tf.zeros((1, board_h, board_w, input_channels))
        self.q_network(dummy)
        self.target_network(dummy)
        self.target_network.set_weights(self.q_network.get_weights())

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter
        self.train_step = 0

    @property
    def epsilon(self):
        """Current epsilon value (exponential decay)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, state, action_mask=None):
        """
        Select actions using ε-greedy policy.

        Args:
            state: numpy array (n_boards, H, W, C) — channels-last
            action_mask: numpy array (n_boards, 4) with 1=valid, 0=invalid

        Returns:
            actions: numpy array (n_boards,)
        """
        n_boards = state.shape[0]
        eps = self.epsilon
        self.steps_done += 1

        # Decide which boards explore vs exploit
        explore = np.random.rand(n_boards) < eps

        actions = np.zeros(n_boards, dtype=np.int64)

        # Exploit: use Q-network
        exploit_idx = np.where(~explore)[0]
        if len(exploit_idx) > 0:
            state_tensor = tf.constant(state[exploit_idx], dtype=tf.float32)
            q_values = self.q_network(state_tensor)

            # Apply action mask
            if action_mask is not None and self.use_action_mask:
                mask_tensor = tf.constant(action_mask[exploit_idx], dtype=tf.float32)
                q_values = q_values + (1.0 - mask_tensor) * (-1e8)

            actions[exploit_idx] = tf.argmax(q_values, axis=-1).numpy()

        # Explore: random valid action
        explore_idx = np.where(explore)[0]
        if len(explore_idx) > 0:
            if action_mask is not None and self.use_action_mask:
                for i in explore_idx:
                    valid = np.where(action_mask[i] > 0)[0]
                    if len(valid) > 0:
                        actions[i] = np.random.choice(valid)
                    else:
                        actions[i] = np.random.randint(0, self.n_actions)
            else:
                actions[explore_idx] = np.random.randint(0, self.n_actions, size=len(explore_idx))

        return actions

    def store_transition(self, states, actions, rewards, next_states,
                         masks=None, next_masks=None):
        """Store a batch of transitions in the replay buffer."""
        self.replay_buffer.push_batch(states, actions, rewards, next_states,
                                       masks, next_masks)

    def train_step_fn(self):
        """Perform one training step (mini-batch SGD on replay buffer)."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, masks, next_masks = \
            self.replay_buffer.sample(self.batch_size)

        states_t = tf.constant(states, dtype=tf.float32)
        actions_t = tf.constant(actions, dtype=tf.int64)
        rewards_t = tf.constant(rewards, dtype=tf.float32)
        next_states_t = tf.constant(next_states, dtype=tf.float32)

        # Always pass a mask tensor to avoid @tf.function retracing
        if next_masks is not None and self.use_action_mask:
            nm_t = tf.constant(next_masks, dtype=tf.float32)
        else:
            nm_t = tf.ones((self.batch_size, self.n_actions), dtype=tf.float32)

        loss = self._train_step_compiled(states_t, actions_t, rewards_t, next_states_t, nm_t)

        # Periodically update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._update_target_network()

        return loss.numpy()

    @tf.function
    def _train_step_compiled(self, states_t, actions_t, rewards_t, next_states_t, nm_t):
        """Compiled training step (tf.function for graph-mode speed)."""
        # Double DQN: select best action with online net, evaluate with target net
        next_q_online = self.q_network(next_states_t, training=False)
        next_q_online = next_q_online + (1.0 - nm_t) * (-1e8)
        best_actions = tf.argmax(next_q_online, axis=-1)

        next_q_target = self.target_network(next_states_t, training=False)
        next_q = tf.gather(next_q_target, best_actions, batch_dims=1)

        target = rewards_t + self.gamma * next_q

        # Gradient step
        with tf.GradientTape() as tape:
            q_values = self.q_network(states_t, training=True)
            batch_size = tf.shape(actions_t)[0]
            action_indices = tf.stack([tf.cast(tf.range(batch_size), tf.int64), actions_t], axis=1)
            q_selected = tf.gather_nd(q_values, action_indices)
            loss = tf.reduce_mean(tf.square(q_selected - target))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        return loss

    def _update_target_network(self):
        """Polyak averaging: θ_target = τ·θ + (1-τ)·θ_target."""
        for target_var, var in zip(self.target_network.trainable_variables, self.q_network.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)

    def save(self, filepath):
        """Save Q-network weights."""
        self.q_network.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath, eval_mode=True):
        """Load Q-network weights and sync target network."""
        self.q_network.load_weights(filepath)
        self.target_network.set_weights(self.q_network.get_weights())
        print(f"Model loaded from {filepath}")

    def select_action_greedy(self, state, action_mask=None):
        """Select actions greedily (no exploration) — for evaluation."""
        state_tensor = tf.constant(state, dtype=tf.float32)
        q_values = self.q_network(state_tensor)

        if action_mask is not None and self.use_action_mask:
            mask_tensor = tf.constant(action_mask, dtype=tf.float32)
            q_values = q_values + (1.0 - mask_tensor) * (-1e8)

        return tf.argmax(q_values, axis=-1).numpy()
