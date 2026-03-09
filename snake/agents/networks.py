"""
Shared CNN network architectures for Snake RL agents.
Supports both fully observable (7x7x4) and partially observable (5x5x4) inputs.
"""
import tensorflow as tf


class SnakeCNN(tf.keras.Model):
    """
    Base CNN feature extractor for Snake.
    Input: (batch, H, W, channels=4) — one-hot encoded board (channels-last)
    Output: (batch, 128) feature vector
    """

    def __init__(self, input_channels=4, board_h=7, board_w=7):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(128, activation='relu')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class QNetwork(tf.keras.Model):
    """
    Q-Network for DQN.
    Input: state (batch, H, W, C)
    Output: Q-values for all actions (batch, n_actions=4)
    """

    def __init__(self, input_channels=4, board_h=7, board_w=7, n_actions=4):
        super().__init__()
        self.features = SnakeCNN(input_channels, board_h, board_w)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.q_out = tf.keras.layers.Dense(n_actions)

    def call(self, x):
        features = self.features(x)
        x = self.dense1(features)
        return self.q_out(x)


class PolicyNetwork(tf.keras.Model):
    """
    Policy Network for REINFORCE.
    Input: state (batch, H, W, C)
    Output: action probabilities (batch, n_actions=4)
    """

    def __init__(self, input_channels=4, board_h=7, board_w=7, n_actions=4):
        super().__init__()
        self.features = SnakeCNN(input_channels, board_h, board_w)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.logits_out = tf.keras.layers.Dense(n_actions)

    def call(self, x, action_mask=None):
        features = self.features(x)
        logits = self.logits_out(self.dense1(features))

        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e8)

        return tf.nn.softmax(logits, axis=-1)

    def get_logits(self, x, action_mask=None):
        """Return raw logits (for log_softmax)."""
        features = self.features(x)
        logits = self.logits_out(self.dense1(features))

        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e8)

        return logits


class ValueNetwork(tf.keras.Model):
    """
    Value Network (baseline / critic).
    Input: state (batch, H, W, C)
    Output: state value V(s) (batch, 1)
    """

    def __init__(self, input_channels=4, board_h=7, board_w=7):
        super().__init__()
        self.features = SnakeCNN(input_channels, board_h, board_w)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.value_out = tf.keras.layers.Dense(1)

    def call(self, x):
        features = self.features(x)
        return self.value_out(self.dense1(features))


class ActorCriticNetwork(tf.keras.Model):
    """
    Combined Actor-Critic Network (shared CNN backbone).
    Input: state (batch, H, W, C)
    Output: (action_probs, state_value)
    """

    def __init__(self, input_channels=4, board_h=7, board_w=7, n_actions=4):
        super().__init__()
        self.features = SnakeCNN(input_channels, board_h, board_w)
        # Policy head
        self.policy_dense = tf.keras.layers.Dense(64, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(n_actions)
        # Value head
        self.value_dense = tf.keras.layers.Dense(64, activation='relu')
        self.value_out = tf.keras.layers.Dense(1)

    def call(self, x, action_mask=None):
        features = self.features(x)

        logits = self.policy_logits(self.policy_dense(features))
        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e8)
        probs = tf.nn.softmax(logits, axis=-1)

        value = self.value_out(self.value_dense(features))

        return probs, value

    def get_logits_and_value(self, x, action_mask=None):
        """Return raw logits and value (for training)."""
        features = self.features(x)

        logits = self.policy_logits(self.policy_dense(features))
        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e8)

        value = self.value_out(self.value_dense(features))

        return logits, value
