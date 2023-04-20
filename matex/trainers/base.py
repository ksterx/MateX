import tempfile
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
from matex.envs import MatexEnv, VecMatexEnv, env_name_aliases, get_metrics_dict, get_reward_func

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

        self.render_mode = "human" if cfg.render else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()  # TODO: use cfg.num_gpus
        self.env = self._make_env(num_envs=cfg.num_envs)

        # Set agent(s)
        self._set_agent()

    def train(self):
        self._set_logger()

        num_episodes = self.cfg.num_episodes if not self.cfg.debug else 3
        with tempfile.TemporaryDirectory() as temp_dir:
            with trange(num_episodes) as pbar:
                best_metric_ep = -float("inf")
                for ep in pbar:
                    pbar.set_description(f"[TRAIN] Episode: {ep+1:>5}")
                    best_metric_step = -float("inf")

                    state, _ = self.env.reset()

                    for step in range(self.cfg.max_steps):
                        action = [
                            agent.act.remote(
                                state[i],
                                eps=self.cfg.eps_max,
                                prog_rate=ep / self.cfg.num_episodes,
                                eps_decay=self.cfg.eps_decay,
                                eps_min=self.cfg.eps_min,
                            )
                            for i, agent in enumerate(self.agents)
                        ]  # TODO: Check order of state

                        action = ray.get(action)

                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        reward = get_reward_func[self.cfg.exp_name](
                            reward=reward,
                            next_state=next_state,
                            terminated=terminated,
                            step=step,
                            max_steps=self.cfg.max_steps,
                            device=self.device,
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

    def test(self, num_episodes: int = 3):
        GIF_NAME = "test.gif"
        with tempfile.TemporaryDirectory() as temp_dir:
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
            env = MatexEnv(env, self.device, record=True)
            state, _ = env.reset()
            terminated, truncated = False, False

            # Play episodes
            with trange(num_episodes) as pbar:
                for ep in pbar:
                    pbar.set_description(f"[TEST] Episode: {ep+1:>5}")
                    while not (terminated or truncated):
                        action = self.agent.act.remote(state, deterministic=True)
                        action = ray.get(action)
                        state, _, terminated, truncated, _ = env.step(action)

            # Save gif
            env.save_gif(temp_dir, GIF_NAME)
            self.logger.log_artifact(f"{temp_dir}/{GIF_NAME}")

    def play(self, num_episodes: int = 3):
        # Prepare environment
        env = gym.make(self.env_name, render_mode="human")
        env = MatexEnv(env, self.device)
        state, _ = env.reset()
        terminated, truncated = False, False

        # Play episodes
        with trange(num_episodes) as pbar:
            for ep in pbar:
                pbar.set_description(f"[PLAY] Episode: {ep+1:>5}")
                while not (terminated or truncated):
                    action = self.agent.act.remote(state, deterministic=True)
                    action = ray.get(action)
                    state, _, terminated, truncated, _ = env.step(action)

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
        if int(num_envs) == 1:
            env = gym.make(self.env_name, render_mode=render_mode)
            env = MatexEnv(env, self.device)
            return env
        elif int(num_envs) > 1:
            env = gym.vector.make(
                self.env_name, render_mode=render_mode, num_envs=self.cfg.num_envs
            )
            env = VecMatexEnv(env, self.device)
            return env
        else:
            raise ValueError(f"Invalid number of environments: {num_envs}")

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
