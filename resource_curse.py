import random
import numpy as np


class ResourceCurseEnv:
    def __init__(
        self, reward=5, social_reward=2, alpha=1.5, political_reward=1, tran_coef=0.05
    ):

        self.state = 0.5
        self.time = 0

        self.reward = reward
        self.social_reward = social_reward
        self.alpha = alpha
        self.political_reward = political_reward
        self.tran_coef = tran_coef

        self.action_space = np.array([0, 1])
        self.obs_shape = (1,)

    def reset(self):
        self.time = 0
        self.state = random.random()
        return self.state

    def get_reward(self, state, action):
        action_country, action_world = action

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
        action_country, action_world = action

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

        obs = self._clamp(self.next_state(self.state, action))

        # if self.state >= 0.5 and self.state < 0.55:
        #     print("-" * 10)
        #     print(f"observating: {obs} from ({self.state}, {action})")
        #     print(f"reward: {reward}")

        self.state = obs
        self.time += 1
        done = self.time > 200

        return obs, reward, done, None

    def _clamp(self, val, minv=0, maxv=0.99):
        return max(minv, min(maxv, val))
