import gymnasium as gym
import numpy as np


class Agent:
    """
    Custom Agent RL
    """
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        # Set hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize Q-table
        self.q_table = np.zeros((observation_space.shape[0], action_space.n))

    def act(self, observation: np.ndarray) -> int:
        """
        Acts
        """
        if np.random.uniform() < self.epsilon:
            # Choose random action
            return self.action_space.sample()
        else:
            # Choose action with highest Q-value
            return np.argmax(self.q_table[observation])

    def learn(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Learns
        """
        # Update Q-value for current state-action pair
        current_q = self.q_table[observation, action]
        next_max_q = np.max(self.q_table[next_observation])
        td_error = reward + self.discount_factor * next_max_q * (not done) - current_q
        self.q_table[observation, action] += self.learning_rate * td_error

        # Update epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
