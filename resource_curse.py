import random
import numpy as np


class ResourceCurseEnv:
    def __init__(
        self, reward=5, social_reward=2, alpha=1.5, political_reward=1, tran_coef=0.05
    ):
        # init env
        self.state = 0.5
        self.time = 0

        # init parameters
        self.reward = reward
        self.social_reward = social_reward
        self.alpha = alpha
        self.political_reward = political_reward
        self.tran_coef = tran_coef

        # init env attributes
        self.action_space = np.array([0, 1])
        self.obs_shape = (1,)

    def reset(self):
        "restart environment with a random initial state"
        self.time = 0
        self.state = random.random()
        return self.state

    def get_reward(self, state, action):
        "reward calculation"

        # get actions of co-players
        action_country, action_world = action

        # determine quadrant in reward table and calculate rewards
        if action_country and action_world:
            return (
                -state * self.social_reward,
                -(state - 0.5) * 2 * self.political_reward,
            )
        elif action_country and not action_world:
            return (state * self.alpha * self.reward, self.alpha * self.reward)
        elif not action_country and action_world:
            return (state * self.social_reward, -(state - 0.5) * self.political_reward)
        elif not action_country and not action_world:
            return (state * self.reward, self.reward)

    def next_state(self, state, action):
        # get actions of co-players
        action_country, action_world = action

        # determine next state based on actions
        if action_country and action_world:
            return state - self.tran_coef
        elif action_country and not action_world:
            return state - (self.tran_coef / 2)
        elif not action_country and action_world:
            return state + (self.tran_coef / 2)
        elif not action_country and not action_world:
            return state + self.tran_coef

    def step(self, action):
        reward = self.get_reward(self.state, action)

        # the self._clamp ensures the next states is between 0 and 1
        obs = self._clamp(self.next_state(self.state, action))

        self.state = obs
        self.time += 1
        done = self.time > 200  # for now, the stop condition is a time limit

        return obs, reward, done, None

    def _clamp(self, val, minv=0, maxv=0.99):
        return max(minv, min(maxv, val))
