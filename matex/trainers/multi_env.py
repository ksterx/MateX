import tempfile
from typing import Dict, List, Optional, Union

import gymnasium as gym
import ray
from tqdm import tqdm

from matex import Experience
from matex.agents import DQN
from matex.envs import RayEnv, get_metrics_dict, get_reward_func

from .base import Trainer


class MultiEnvTrainer(Trainer):
    def __init__(self, cfg, callbacks=None, logger=None):
        super().__init__(cfg, callbacks, logger)

    def train(self):
        self._set_logger()

        num_episodes = self.cfg.num_episodes if not self.cfg.debug else 10
        ep = 0
        pbar = tqdm(total=num_episodes)

        pbar.update()
        pbar.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            while ep < num_episodes:
                best_metric_ep = -float("inf")

                pbar.set_description(f"[TRAIN] Episode: {ep+1:>5}")
                best_metric_step = -float("inf")

                state, info = self.env.reset()

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
                    reward = [
                        get_reward_func[self.cfg.exp_name].remote(
                            r,
                            t,
                            step,
                            self.cfg.max_steps,
                            self.device,
                        )
                        for r, t in zip(reward, terminated)
                    ]
                    metrics, metric_name = get_metrics_dict[self.cfg.exp_name].remote(
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

    def _make_env(
        self,
        num_envs: int = 4,
        render_mode: str = "human",
    ) -> Union[gym.Env, gym.vector.VectorEnv]:
        # Prepare environment for training
        env = gym.make(self.env_name, render_mode=render_mode)
        env_obj = [RayEnv.remote(env, self.device) for _ in range(num_envs)]
        return env, env_obj

    def _set_agent(self):
        self.agents = [
            DQN.options(num_gpus=self.num_gpus / self.cfg.num_envs).remote(
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
