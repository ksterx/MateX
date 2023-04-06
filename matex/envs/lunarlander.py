def get_reward_lunarlander(reward, terminated, step, max_steps, device):
    return reward


def get_metrics_lunarlander(state, reward, step):
    return {"reward": reward.item()}, "reward"
