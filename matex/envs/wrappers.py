from typing import List, Union

import gymnasium as gym
import numpy as np
import ray
import torch

# from gymnasium.vector import VectorEnvWrapper
from matplotlib import animation, pyplot


@ray.remote
class RayEnv(gym.Wrapper):
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
        )  # TODO: Fix this
        patch = pyplot.imshow(self.frames[0])
        pyplot.axis("off")

        def animate(i):
            patch.set_data(self.frames[i])

        anim = animation.FuncAnimation(pyplot.gcf(), animate, frames=len(self.frames), interval=50)
        anim.save(f"{dir_path}/{filename}", writer="imagemagick", fps=60)


# class VecMatexEnv(VectorEnvWrapper):
#     def __init__(self, env, device, **kwargs):
#         super().__init__(env, **kwargs)
#         self.device = device

#     def step(self, action: List[torch.Tensor]):
#         action = self.list_tensor2array(action)
#         next_state, reward, terminated, truncated, info = self.env.step(action)
#         next_state = [
#             torch.tensor(next_state[i], device=self.device, dtype=torch.float).view(1, -1)
#             for i in range(self.env.num_envs)
#         ]
#         reward = [
#             torch.tensor(reward[i], device=self.device, dtype=torch.float).view(-1, 1)
#             for i in range(self.env.num_envs)
#         ]
#         print(f"next_states: {next_state}")
#         print(f"rewards: {reward}")
#         print(f"terminated: {terminated}")
#         print(f"truncated: {truncated}")
#         print(f"infos: {info}")

#         done = terminated or truncated


#         return next_state, reward, terminated, truncated, info

#     def reset(self):
#         state, info = self.env.reset()
#         print(f"state: {state} shape: {state.shape}")
#         state = [
#             torch.tensor(state[i], device=self.device, dtype=torch.float).view(1, -1)
#             for i in range(self.env.num_envs)
#         ]
#         info["step"] = [0 for _ in range(self.env.num_envs)]
#         return state, info

#     def list_tensor2array(self, list_tensor: List[torch.Tensor]) -> np.ndarray:
#         return np.array([t.squeeze().cpu().numpy() for t in list_tensor])

#     def on_step_end(self):
