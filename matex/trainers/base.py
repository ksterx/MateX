import tempfile
from typing import Dict, List, Optional, Union

import gymnasium as gym
import ray
import torch
from omegaconf import DictConfig
from tqdm import trange

from matex import Experience
from matex.agents import DQN
from matex.common import Callback, notice
from matex.common.loggers import MLFlowLogger
from matex.envs import EnvWrapper, env_name_aliases, get_metrics_dict, get_reward_dict

# import heartrate

# heartrate.trace(browser=True)


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

        render_mode = "human" if cfg.render else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()  # TODO: use cfg.num_gpus
        env = gym.make(self.env_name, render_mode=render_mode)
        self.env = EnvWrapper(env, self.device)

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
        )

    def train(self):
        if self.logger == "mlflow":
            self.logger = MLFlowLogger(tracking_uri=self.cfg.mlflow_uri, cfg=self.cfg)
            self.logger.log_hparams(self.cfg)

        n_episodes = self.cfg.n_episodes if not self.cfg.debug else 10
        with tempfile.TemporaryDirectory() as temp_dir:
            with trange(n_episodes) as pbar:
                best_metric_ep = -float("inf")
                for ep in pbar:
                    pbar.set_description(f"[TRAIN] Episode: {ep+1:>5}")
                    best_metric_step = -float("inf")

                    state, _ = self.env.reset()

                    for step in range(self.cfg.max_steps):
                        action = self.agent.act.remote(
                            state,
                            eps=self.cfg.eps_max,
                            prog_rate=ep / self.cfg.n_episodes,
                            eps_decay=self.cfg.eps_decay,
                            eps_min=self.cfg.eps_min,
                        )

                        action = ray.get(action)

                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        reward = get_reward_dict[self.cfg.exp_name](
                            reward,
                            terminated,
                            step,
                            self.cfg.max_steps,
                            self.device,
                        )
                        metrics, metric_name = get_metrics_dict[self.cfg.exp_name](
                            state=state,
                            reward=reward,
                            step=step,
                        )
                        if metrics[metric_name] > best_metric_step:
                            best_metric_step = metrics[metric_name]

                        self.logger.log_metrics(metrics=metrics, step=step, prefix="step_")

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
                                key=metric_name, value=best_metric_step, step=ep, prefix="episode_"
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

    def test(self, n_episodes=3):
        import gym
        import moviepy.editor as mpy
        from gym.wrappers import RecordVideo

        with tempfile.TemporaryDirectory() as temp_dir:
            self.load(
                f"./experiments/results/{self.logger.experiment_id}/{self.logger.run_id}/artifacts/best.ckpt"
            )
            env = gym.make(self.env_name, render_mode="rgb_array")
            env = RecordVideo(env, f"{temp_dir}")

            state, _ = env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)
            terminated, truncated = False, False

            with trange(n_episodes) as pbar:
                for ep in pbar:
                    pbar.set_description(f"[TEST] Episode: {ep+1:>5}")
                    while not (terminated or truncated):
                        action = self.agent.act.remote(state, deterministic=True)
                        action = ray.get(action)
                        state, _, terminated, truncated, _ = env.step(action.item())
                        state = torch.tensor(
                            state,
                            device=self.device,
                            dtype=torch.float,
                        ).view(1, -1)

            movie = mpy.VideoFileClip(f"{temp_dir}/rl-video-episode-0.mp4")
            movie.write_gif(f"{temp_dir}/result.gif")

            self.logger.log_artifact(f"{temp_dir}/result.gif")

    def play(self, n_episodes=3):
        env = gym.make(self.env_name, render_mode="human")
        env = EnvWrapper(env, self.device)
        state, _ = env.reset()
        terminated, truncated = False, False

        with trange(n_episodes) as pbar:
            for ep in pbar:
                pbar.set_description(f"[PLAY] Episode: {ep+1:>5}")
                while not (terminated or truncated):
                    action = self.agent.act.remote(state, deterministic=True)
                    action = ray.get(action)
                    state, _, terminated, truncated, _ = env.step(action)

    def load(self, ckpt_path):
        self.agent.load.remote(ckpt_path)
