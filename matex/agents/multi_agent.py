from typing import List

from .base import Agent


class MultiAgents:
    def __init__(self, agents: List[Agent]) -> None:
        self.agents = [
            agents.options(num_gpus=self.num_gpus / self.cfg.num_envs).remote(
                lr=self.acfg.lr,
                gamma=self.acfg.gamma,
                memory_size=self.acfg.memory_size,
                batch_size=self.acfg.batch_size,
                state_size=self.env.observation_space.shape[0],
                action_size=self.env.action_space.nvec[0],  # TODO: only works for Discrete
                hidden_size=self.acfg.hidden_size,
                device=self.device,
                is_ddqn=self.acfg.is_ddqn,
                id=i,
            )
            for i in range(self.cfg.num_envs)
        ]
