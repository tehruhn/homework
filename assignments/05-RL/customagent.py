import gymnasium as gym
import numpy as np


class Agent:
    """
    Custom Gym Agent
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.min_epsilon = 0.01

        self.num_buckets = (20, 20, 20, 20, 20, 20, 2, 2)
        self.q_table = np.zeros(self.num_buckets + (action_space.n,))

    def discretize(self, observation: gym.spaces.Box) -> tuple:
        """
        Discretizes
        """
        lower_bounds = self.observation_space.low
        upper_bounds = self.observation_space.high
        lower_bounds[6] = -1
        upper_bounds[6] = 1
        lower_bounds[7] = -1
        upper_bounds[7] = 1

        ratios = [
            (observation[i] + abs(lower_bounds[i]))
            / (upper_bounds[i] - lower_bounds[i])
            for i in range(len(observation))
        ]
        new_observation = [
            int(round((self.num_buckets[i] - 1) * ratios[i]))
            for i in range(len(observation))
        ]
        new_observation = np.array(new_observation)
        new_observation = np.minimum(
            self.num_buckets - 1, np.maximum(0, new_observation)
        )

        return tuple(new_observation)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Acts
        """
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            discretized_obs = self.discretize(observation)
            return np.argmax(self.q_table[discretized_obs])

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Learns
        """
        discretized_obs = self.discretize(observation)
        action = self.act(observation)
        new_value = reward + self.gamma * np.max(self.q_table[discretized_obs])

        old_value = self.q_table[discretized_obs + (action,)]
        updated_value = old_value + self.alpha * (new_value - old_value)
        self.q_table[discretized_obs + (action,)] = updated_value

        if terminated or truncated:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)
