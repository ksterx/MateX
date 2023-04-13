import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnvWrapper
from matplotlib import animation, pyplot


class MatexEnv(gym.Wrapper):
    def __init__(self, env, device, record=False, **kwargs):
        super().__init__(env, **kwargs)
        self.device = device
        self.record = record
        self.frames = []

    def step(self, action):
        action = action.item()
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float).view(1, -1)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float).view(1, 1)
        if self.record:
            self.frames.append(self.env.render())
        return next_state, reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()
        state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)
        return state, info

    def save_gif(self, dir_path: str, filename: str):
        pyplot.figure(
            figsize=(self.frames[0].shape[1] / 72.0, self.frames[0].shape[0] / 72.0), dpi=72
        )
        patch = pyplot.imshow(self.frames[0])
        pyplot.axis("off")

        def animate(i):
            patch.set_data(self.frames[i])

        anim = animation.FuncAnimation(pyplot.gcf(), animate, frames=len(self.frames), interval=50)
        anim.save(f"{dir_path}/{filename}", writer="imagemagick", fps=60)


class VecMatexEnv(VectorEnvWrapper):
    def __init__(self, env, device, **kwargs):
        super().__init__(env, **kwargs)
        self.device = device

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = [
            torch.tensor(next_state[i], device=self.device, dtype=torch.float).view(1, -1)
            for i in range(self.env.num_envs)
        ]
        reward = [
            torch.tensor(reward[i], device=self.device, dtype=torch.float).view(-1, 1)
            for i in range(self.env.num_envs)
        ]
        print(f"next_states: {next_state} shape: {next_state.shape}")
        print(f"rewards: {reward} shape: {reward.shape}")
        print(f"terminated: {terminated} shape: {terminated.shape}")
        print(f"truncated: {truncated} shape: {truncated.shape}")
        print(f"infos: {info}")
        return next_state, reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()
        state = [
            torch.tensor(state[i], device=self.device, dtype=torch.float).view(1, -1)
            for i in range(self.env.num_envs)
        ]
        print(f"states: {state}")
        print(f"stage shape: {len(state)}")
        print(f"infos: {info}")
        print(f"info shape: {len(info)}")
        return state, info
