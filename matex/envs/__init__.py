from .acrobot import get_metrics_acrobot, get_reward_acrobot
from .cartpole import get_metrics_cartpole, get_reward_cartpole
from .lunarlander import get_metrics_lunarlander, get_reward_lunarlander
from .mountaincar import get_metrics_mountaincar, get_reward_mountaincar
from .wrappers import RayEnv

get_reward_func = {
    "cartpole": get_reward_cartpole,
    "acrobot": get_reward_acrobot,
    "mountaincar": get_reward_mountaincar,
    "lunarlander": get_reward_lunarlander,
}

get_metrics_dict = {
    "cartpole": get_metrics_cartpole,
    "acrobot": get_metrics_acrobot,
    "mountaincar": get_metrics_mountaincar,
    "lunarlander": get_metrics_lunarlander,
}

env_name_aliases = {
    "cartpole": "CartPole-v1",
    "acrobot": "Acrobot-v1",
    "mountaincar": "MountainCar-v0",
    "lunarlander": "LunarLander-v2",
}
