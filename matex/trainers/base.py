import os
import tempfile
import time
from pathlib import Path
from typing import List, Union

import gymnasium as gym
import ray
import torch
from omegaconf import DictConfig
from tqdm import trange

from matex import Experience
from matex.agents import DQN
from matex.common import Callback, notice
from matex.common.loggers import MLFlowLogger
from matex.envs import (RayEnv, env_name_aliases, get_metrics_dict,
                        get_reward_func)


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        callbacks: List[Callback] = None,
        logger: str = None,
    ) -> None:
        """Manage training process.

        Args:
            cfg (DictConfig): Configurations.
            callbacks (List[Callback], optional): Callback list. Defaults to None.
            logger (str, optional): Logger. Defaults to None.
        """

        self.cfg = cfg
        self.acfg = cfg.agents
        self.callbacks = callbacks
        self.logger = logger
        try:
            self.env_name = env_name_aliases[cfg.exp_name]
        except KeyError:
            notice.error(f"'{cfg.exp_name}' is invalid experiment name")
            raise ValueError(
                f"Registerd envs at matex.envs.env_name_aliases: {[k for k in env_name_aliases.keys()]}"
            )

        self.render_mode = "human" if cfg.render else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()  # TODO: use cfg.num_gpus
        self.env, self.env_ref = self._make_env(num_envs=cfg.num_envs, render_mode=self.render_mode)

        # Set agent(s)
        self._set_agent()

    def run(self):
        self._set_logger()

        num_episodes = self.cfg.num_episodes if not self.cfg.debug else 3
        with tempfile.TemporaryDirectory() as temp_dir:
            with trange(num_episodes) as pbar:
                best_metric_ep = -float("inf")
                for ep in pbar:
                    pbar.set_description(f"[TRAIN] Episode: {ep+1:>5}")
                    best_metric_step = -float("inf")

                    state, _ = ray.get(self.env_ref.reset.remote())

                    for step in range(self.cfg.max_steps):
                        action, info = ray.get(
                            self.agent.act.remote(
                                state,
                                eps=self.cfg.eps_max,
                                prog_rate=ep / self.cfg.num_episodes,
                                eps_min=self.cfg.eps_min,
                            )
                        )

                        next_state, reward, terminated, truncated, info = ray.get(
                            self.env_ref.step.remote(action.cpu())
                        )

                        reward = ray.get(
                            get_reward_func[self.cfg.exp_name].remote(
                                reward=reward,
                                next_state=next_state,
                                terminated=terminated,
                                step=step,
                                max_steps=self.cfg.max_steps,
                                device=torch.device("cpu"),
                            )
                        )
                        metrics, metric_name = ray.get(
                            get_metrics_dict[self.cfg.exp_name].remote(
                                state=state,
                                reward=reward,
                                step=step,
                            )
                        )
                        if metrics[metric_name] > best_metric_step:
                            best_metric_step = metrics[metric_name]

                        self.logger.log_metrics(
                            metrics=metrics, step=step, prefix="step_"
                        )

                        exp = Experience(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state,
                            terminated=terminated,
                            truncated=truncated,
                            info=info,
                        )
                        self.agent.memorize.remote(experience=exp)
                        self.agent.learn.remote()

                        state = next_state

                        if terminated or truncated:
                            self.logger.log_metric(
                                key=metric_name,
                                value=best_metric_step,
                                step=ep,
                                prefix="episode_",
                            )
                            self.agent.save.remote(f"{temp_dir}/chekpoint.ckpt")
                            if best_metric_step >= best_metric_ep:
                                best_metric_ep = best_metric_step
                                self.agent.save.remote(f"{temp_dir}/best.ckpt")
                            break

                        self.agent.on_step_end.remote(step, **self.acfg)

                    pbar.set_postfix(metric=f"{best_metric_step:.3g}")

            self.logger.log_artifact(f"{temp_dir}/best.ckpt")
            self.logger.log_artifact(f"{temp_dir}/chekpoint.ckpt")
        self.logger.close()

    def test(self, num_episodes: int = 3):
        GIF_NAME = "test.gif"

        # Load best checkpoint
        try:
            self.load(
                Path().cwd()
                / Path("experiments/results")
                / self.logger.experiment_id
                / self.logger.run_id
                / Path("artifacts/best.ckpt")
            )
        except FileNotFoundError as e:
            notice.error("Could not find best.ckpt file in artifacts directory")
            raise e

        # Prepare environment
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = RayEnv.options(num_cpus=1).remote(env, self.device, record=True)
        state, _ = ray.get(env.reset.remote())
        terminated, truncated = False, False

        # Play episodes
        with trange(num_episodes) as pbar:
            for ep in pbar:
                pbar.set_description(f"[TEST] Episode: {ep+1:>5}")
                while not (terminated or truncated):
                    action, _ = ray.get(
                        self.agent.act.remote(state, deterministic=True)
                    )
                    state, _, terminated, truncated, _ = ray.get(
                        env.step.remote(action.cpu())
                    )

        # Save gif (DO NOT USE Tempfile)
        env.save_gif.remote(os.getcwd(), GIF_NAME)
        time.sleep(1)
        self.logger.log_artifact(GIF_NAME)

    def play(self, num_episodes: int = 3):
        # Prepare environment
        env = gym.make(self.env_name, render_mode="human")
        env = RayEnv.options(num_cpus=1).remote(env, self.device)
        state, _ = ray.get(env.reset.remote())
        terminated, truncated = False, False

        # Play episodes
        with trange(num_episodes) as pbar:
            for ep in pbar:
                pbar.set_description(f"[PLAY] Episode: {ep+1:>5}")
                while not (terminated or truncated):
                    action, _ = ray.get(
                        self.agent.act.remote(state, deterministic=True)
                    )
                    state, _, terminated, truncated, _ = ray.get(
                        env.step.remote(action.cpu())
                    )

    def load(self, ckpt_path):
        self.agent.load.remote(ckpt_path)

    def _set_logger(self):
        # Convert logger name to logger object
        if self.logger == "mlflow":
            self.logger = MLFlowLogger(tracking_uri=self.cfg.mlflow_uri, cfg=self.cfg)
            self.logger.log_hparams(self.cfg)

    def _make_env(
        self,
        num_envs: int = 1,
        render_mode: str = "human",
    ) -> Union[gym.Env, gym.vector.VectorEnv]:
        # Prepare environment for training
        env = gym.make(self.env_name, render_mode=render_mode)
        env_ref = RayEnv.options(num_cpus=1).remote(env, torch.device("cpu"))
        return env, env_ref

    def _set_agent(self):
        self.agent = DQN.options(num_gpus=self.num_gpus).remote(
            lr=self.acfg.lr,
            gamma=self.acfg.gamma,
            memory_size=self.acfg.memory_size,
            batch_size=self.acfg.batch_size,
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            hidden_size=self.acfg.hidden_size,
            device=self.device,
            is_ddqn=self.acfg.is_ddqn,
            id=0,
        )
