import gymnasium as gym
import numpy as np
import torch


class EnvWrapper(gym.Wrapper):
    def __init__(self, env, device, **kwargs):
        super().__init__(env, **kwargs)
        self.device = device

    def step(self, action):
        action = action.item()
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float).view(1, -1)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float).view(1, 1)
        return next_state, reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()
        state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)
        return state, info
