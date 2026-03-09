"""
Actor-Critic Agent for Snake.
Shared backbone with separate policy (actor) and value (critic) heads.
Uses n-step returns with bootstrapping for the non-episodic environment.
"""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.networks import ActorCriticNetwork


class ActorCriticAgent:
    """
    n-step Actor-Critic with shared backbone.

    Unlike REINFORCE (which uses full Monte Carlo returns), Actor-Critic
    bootstraps value estimates, trading some bias for lower variance.
    The shared CNN backbone feeds into separate policy and value heads.
    """

    def __init__(self, input_channels=4, board_h=7, board_w=7, n_actions=4,
                 lr=5e-4, gamma=0.95, use_action_mask=True,
                 entropy_coef=0.01, n_steps=5):

        self.n_actions = n_actions
        self.gamma = gamma
        self.use_action_mask = use_action_mask
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps

        # Shared Actor-Critic network
        self.network = ActorCriticNetwork(
            input_channels, board_h, board_w, n_actions
        )

        # Build with dummy input (channels-last)
        dummy = tf.zeros((1, board_h, board_w, input_channels))
        self.network(dummy)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def select_action(self, state, action_mask=None):
        """
        Sample action from policy π(a|s; θ).
        Returns actions as a numpy array (n_boards,).
        """
        state_t = tf.constant(state, dtype=tf.float32)
        mask_t = None
        if action_mask is not None and self.use_action_mask:
            mask_t = tf.constant(action_mask, dtype=tf.float32)

        probs, value = self.network(state_t, action_mask=mask_t)

        # Sample from categorical
        logits = tf.math.log(probs + 1e-8)
        actions = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

        return actions.numpy()

    def train_n_steps(self, env, get_mask_fn=None):
        """
        Collect n steps and perform a combined actor-critic update.
        Uses n-step returns with bootstrapped V(s_n) for the advantage.
        """
        saved_states = []
        saved_actions = []
        saved_masks = []
        saved_rewards = []

        for t in range(self.n_steps):
            state = env.to_state()
            if hasattr(state, 'numpy'):
                state = state.numpy()

            mask = None
            if get_mask_fn is not None and self.use_action_mask:
                mask = get_mask_fn(env)

            actions = self.select_action(state, mask)

            rewards = env.move(actions.reshape(-1, 1))
            rewards_np = rewards.numpy().flatten()

            saved_states.append(state)
            saved_actions.append(actions)
            saved_masks.append(mask)
            saved_rewards.append(rewards_np)

        # Bootstrap value for the last state: V(s_n)
        last_state = env.to_state()
        if hasattr(last_state, 'numpy'):
            last_state = last_state.numpy()
        last_state_t = tf.constant(last_state, dtype=tf.float32)
        _, bootstrap_value = self.network(last_state_t)
        bootstrap_value = tf.squeeze(bootstrap_value, axis=-1).numpy()

        # Compute n-step returns (backwards)
        N = saved_rewards[0].shape[0]
        returns_np = np.zeros((self.n_steps, N), dtype=np.float32)
        R = bootstrap_value.copy()
        for t in reversed(range(self.n_steps)):
            R = saved_rewards[t] + self.gamma * R
            returns_np[t] = R

        returns_t = tf.constant(returns_np, dtype=tf.float32)

        # Training step with GradientTape
        with tf.GradientTape() as tape:
            policy_loss = tf.constant(0.0)
            value_loss = tf.constant(0.0)
            entropy_loss = tf.constant(0.0)

            for t in range(self.n_steps):
                state_t = tf.constant(saved_states[t], dtype=tf.float32)
                mask_t = None
                if saved_masks[t] is not None:
                    mask_t = tf.constant(saved_masks[t], dtype=tf.float32)

                logits, value = self.network.get_logits_and_value(state_t, action_mask=mask_t)
                value = tf.squeeze(value, axis=-1)  # (N,)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                probs = tf.nn.softmax(logits, axis=-1)

                # Advantage: A = G_t - V(s_t)
                advantage = tf.stop_gradient(returns_t[t] - value)

                # Gather log probs for chosen actions
                actions_t = tf.constant(saved_actions[t], dtype=tf.int64)
                indices = tf.stack([tf.range(N, dtype=tf.int64), actions_t], axis=1)
                log_prob_selected = tf.gather_nd(log_probs, indices)

                # Actor loss: -log π(a|s) * A
                policy_loss -= tf.reduce_mean(log_prob_selected * advantage)

                # Critic loss: (G_t - V(s_t))²
                value_loss += tf.reduce_mean(tf.square(value - tf.stop_gradient(returns_t[t])))

                # Entropy bonus
                entropy_loss -= tf.reduce_mean(-tf.reduce_sum(probs * log_probs, axis=-1))

            # Total loss
            total_loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

        grads = tape.gradient(total_loss, self.network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        avg_reward = np.mean([np.mean(r) for r in saved_rewards])

        return {
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'avg_reward': avg_reward,
            'entropy': -entropy_loss.numpy() / self.n_steps,
        }

    def select_action_greedy(self, state, action_mask=None):
        """Select the most probable action (for evaluation)."""
        state_t = tf.constant(state, dtype=tf.float32)
        mask_t = None
        if action_mask is not None and self.use_action_mask:
            mask_t = tf.constant(action_mask, dtype=tf.float32)
        probs, _ = self.network(state_t, action_mask=mask_t)
        return tf.argmax(probs, axis=-1).numpy()

    def save(self, filepath):
        """Save model weights."""
        self.network.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath, eval_mode=True):
        """Load model weights."""
        self.network.load_weights(filepath)
        print(f"Model loaded from {filepath}")
