import tempfile

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from matex.agents import DQN
from matex.common import notice
from matex.common.loggers import MLFlowLogger
from matex.envs import EnvWrapper, env_name_aliases, get_metrics_dict, get_reward_dict


class Trainer:
    def __init__(self, cfg, callbacks=None, logger: str = None):

        self.cfg = cfg
        acfg = cfg.agents
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
        env = gym.make(self.env_name, render_mode=render_mode)
        self.env = EnvWrapper(env, self.device)

        self.agent = DQN(
            lr=acfg.lr,
            gamma=acfg.gamma,
            memory_size=acfg.memory_size,
            batch_size=acfg.batch_size,
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            hidden_size=acfg.hidden_size,
            device=self.device,
            is_ddqn=acfg.is_ddqn,
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
                        action = self.agent.act(
                            state,
                            eps=self.cfg.eps_max,
                            prog_rate=ep / self.cfg.n_episodes,
                            eps_decay=self.cfg.eps_decay,
                            eps_min=self.cfg.eps_min,
                        )

                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        reward = get_reward_dict[self.cfg.exp_name](
                            reward,
                            terminated,
                            step,
                            self.cfg.max_steps,
                            self.device,
                        )
                        print(reward)
                        metrics, metric_name = get_metrics_dict[self.cfg.exp_name](
                            state=state,
                            reward=reward,
                            step=step,
                        )
                        if metrics[metric_name] > best_metric_step:
                            best_metric_step = metrics[metric_name]

                        self.logger.log_metrics(metrics=metrics, step=step)

                        self.agent.memorize(state, action, next_state, reward, terminated)
                        self.agent.learn()

                        state = next_state

                        if step % self.cfg.agents.target_net_update_freq == 0:
                            self.agent.update_target_network(
                                soft_update=True,
                                tau=self.cfg.agents.tau,
                            )

                        if terminated or truncated:
                            self.logger.log_metric(metric_name, best_metric_step, step=ep)
                            # self.logger.log_metrics(metrics=metrics, step=ep)
                            self.agent.save(f"{temp_dir}/chekpoint.ckpt")
                            if best_metric_step >= best_metric_ep:
                                best_metric_ep = best_metric_step
                                self.agent.save(f"{temp_dir}/best.ckpt")
                            break

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
                        action = self.agent.act(state, deterministic=True)
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
                    action = self.agent.act(state, deterministic=True)
                    state, _, terminated, truncated, _ = env.step(action)

    def load(self, ckpt_path):
        self.agent.load(ckpt_path)
