import tempfile

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from matex.agents import DQN
from matex.common.loggers import MLFlowLogger


class Trainer:
    def __init__(self, cfg, callbacks=None, logger: str = None):

        self.cfg = cfg
        acfg = cfg.agents
        self.callbacks = callbacks

        if logger == "mlflow":
            self.logger = MLFlowLogger(tracking_uri=cfg.mlflow_uri, cfg=cfg)
            self.logger.log_hparams(cfg)

        render_mode = "human" if cfg.render else None
        self.env = gym.make(cfg.env_name, render_mode=render_mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = DQN(
            lr=acfg.lr,
            gamma=acfg.gamma,
            memory_size=acfg.memory_size,
            batch_size=acfg.batch_size,
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            hidden_size=acfg.hidden_size,
            device=self.device,
        )

    def train(self):
        n_episodes = self.cfg.n_episodes if not self.cfg.debug else 10
        with tempfile.TemporaryDirectory() as temp_dir:
            with trange(n_episodes) as pbar:
                best_step = 0
                for ep in pbar:
                    pbar.set_description(f"[TRAIN] Episode: {ep+1:>5}")

                    state, _ = self.env.reset()
                    state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)

                    for step in range(self.cfg.max_steps):
                        action = self.agent.act(
                            state,
                            eps=self.cfg.eps_max,
                            prog_rate=ep / self.cfg.n_episodes,
                            eps_decay=self.cfg.eps_decay,
                            eps_min=self.cfg.eps_min,
                        )
                        next_state, _, done, _, _ = self.env.step(action.item())
                        next_state = torch.tensor(
                            next_state, device=self.device, dtype=torch.float
                        ).view(1, -1)

                        if done:
                            if step >= self.cfg.max_steps - 5:
                                reward = torch.tensor(
                                    [[1.0]], device=self.device, dtype=torch.float
                                ).view(1, 1)
                            else:
                                reward = torch.tensor(
                                    [[-1.0]], device=self.device, dtype=torch.float
                                ).view(1, 1)
                        else:
                            reward = torch.tensor(
                                [[0.01]], device=self.device, dtype=torch.float
                            ).view(1, 1)

                        self.agent.memorize(state, action, next_state, reward, done)
                        self.agent.learn()

                        state = next_state

                        if step % self.cfg.agents.target_net_update_freq == 0:
                            self.agent.update_target_network(
                                soft_update=True, tau=self.cfg.agents.tau
                            )

                        if done:
                            self.logger.log_metric("step", step + 1, ep)
                            self.agent.save(f"{temp_dir}/chekpoint.ckpt")
                            if step + 1 >= best_step:
                                best_step = step + 1
                                self.agent.save(f"{temp_dir}/best.ckpt")
                            break

                    pbar.set_postfix(step=f"{step+1:*>3}")

            self.logger.log_artifact(f"{temp_dir}/best.ckpt")
            self.logger.log_artifact(f"{temp_dir}/chekpoint.ckpt")

    def test(self, n_episodes=10):
        import gym
        import moviepy.editor as mpy
        from gym.wrappers import RecordVideo

        with tempfile.TemporaryDirectory() as temp_dir:

            self.load(
                f"./experiments/results/{self.logger.experiment_id}/{self.logger.run_id}/artifacts/best.ckpt"
            )
            env = gym.make(self.cfg.env_name, render_mode="rgb_array")
            env = RecordVideo(env, f"{temp_dir}")

            state, _ = env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)
            done = False

            with trange(n_episodes) as pbar:
                for ep in pbar:
                    pbar.set_description(f"[TEST] Episode: {ep+1:>5}")
                    while not done:
                        action = self.agent.act(state, deterministic=True)
                        state, _, done, _, _ = env.step(action.item())
                        state = torch.tensor(state, device=self.device, dtype=torch.float).view(
                            1, -1
                        )

            movie = mpy.VideoFileClip(f"{temp_dir}/rl-video-episode-0.mp4")
            movie.write_gif(f"{temp_dir}/result.gif")

            self.logger.log_artifact(f"{temp_dir}/result.gif")

    def play(self, n_episodes=10):
        env = gym.make(self.cfg.env_name, render_mode="human")
        state, _ = env.reset()
        state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)
        done = False

        for _ in range(n_episodes):
            while not done:
                action = self.agent.act(state, deterministic=True)
                state, _, done, _, _ = env.step(action.item())
                state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)

    def load(self, ckpt_path):
        self.agent.load(ckpt_path)
