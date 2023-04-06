import torch


def get_reward_cartpole(reward, terminated, step, max_steps, device):
    if terminated:
        if step >= max_steps - 5:
            reward = torch.tensor([[1.0]], device=device, dtype=torch.float).view(1, 1)
        else:
            reward = torch.tensor([[-1.0]], device=device, dtype=torch.float).view(1, 1)
    else:
        reward = torch.tensor([[0.01]], device=device, dtype=torch.float).view(1, 1)
    return reward


def get_metrics_cartpole(state, reward, step):
    return {"duration": step + 1}, "duration"
