import math


def get_reward_acrobot(reward, terminated, step, max_steps, device):
    return reward


def get_metrics_acrobot(state, reward, step):
    cos1, sin1, cos2, sin2, _, _ = state.cpu().numpy()[0]
    theta1 = math.atan2(sin1, cos1)
    theta2 = math.atan2(sin2, cos2)
    y = -1 * (math.cos(theta1) + math.cos(theta2 + theta1))
    return {"y": y}, "y"
