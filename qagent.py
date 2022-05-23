import numpy as np
from typing import Any, Tuple

import math

from resource_curse import ResourceCurseEnv


class QAgent:
    def __init__(
        self,
        env: ResourceCurseEnv,
        reward_range: Tuple[int, int],
        discrete_range: int,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 1,
        epsilon_decay: float = 0.995,
    ) -> None:

        # basic parameters
        self.env = env
        self.reward_range = reward_range
        self.discrete_rate = discrete_range
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # assess dimensions of Q-table based off shape of state-action space
        self.action_space_size = len(env.action_space)  # env.action_space.n
        self.obs_shape = env.obs_shape  # env.observation_space.shape
        self.discrete_bucket_shape = self.obs_shape[0] * [discrete_range]
        self.discrete_bucket_size = np.array([1]) / self.discrete_bucket_shape

        # create Q-table
        q_table_shape = tuple(self.discrete_bucket_shape + [self.action_space_size])
        self.q_table = self._initialize_q_table(q_table_shape, reward_range)

    def _initialize_q_table(
        self, q_table_shape: Tuple[int, int, int], reward_range: Tuple[int, int]
    ) -> np.ndarray:
        return np.random.uniform(
            low=reward_range[0], high=reward_range[1], size=q_table_shape
        )

    def _get_discrete_state(self, state: np.ndarray) -> tuple:
        "convert continuous value (0-1) to a discrete value (0-self.obs_shape)"
        return tuple((state / self.discrete_bucket_size).astype(np.int))

    def predict(self, state: np.ndarray, iteration: int) -> np.int:
        "Provide an action given the state, using an epsilon-greedy approach"

        # chance of performing a random action
        # probability decreases with every iteration
        if np.random.random() < self.epsilon * math.pow(self.epsilon_decay, iteration):
            return np.random.randint(0, self.action_space_size)

        # otherwise perform the best possible action given the state
        discrete_state = self._get_discrete_state(state)
        return np.argmax(self.q_table[discrete_state])

    def update(
        self, state: np.ndarray, next_state: np.ndarray, action: np.int, reward: Any
    ):
        "perform q-learning update"

        # get discrete state values
        discrete_state = self._get_discrete_state(state)
        next_discrete_state = self._get_discrete_state(next_state)

        # get current q-value of state-action pair
        # also get best q-value of state that resulted after action was performed
        current_q = self.q_table[discrete_state + (action,)]
        max_next_q = np.max(self.q_table[next_discrete_state])

        # update q-value of state-action pair
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
            reward + self.discount * max_next_q
        )
        self.q_table[discrete_state + (action,)] = new_q

    def hard_update(self, state: np.ndarray, action: np.int, value: np.int):
        "set a q-value, can use this when a 'success' state is reached"
        discrete_state = self._get_discrete_state(state)
        self.q_table[discrete_state + (action,)] = value
