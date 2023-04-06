import gymnasium as gym

from matex.envs.wrappers import VecEnvWrapper

from .base import Trainer


class ParallelTrainer(Trainer):
    def __init__(self, cfg, callbacks=None, logger=None):
        super().__init__(cfg, callbacks, logger)
        self.envs = gym.vector.make(self.env_name, render_mode="human", num_envs=self.cfg.n_envs)

    def train(self):

        self._set_logger(self.logger)

        for ep in range(self.n_episodes):

            states, _ = self.envs.reset()  # (n_envs, state_size)

            # Syncronization method
            for step in range(self.cfg.max_steps):
                actions = self.agent.
