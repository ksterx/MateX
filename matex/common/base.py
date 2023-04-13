import gym
import moviepy.editor as mpy
import torch
from gym.wrappers import RecordVideo
from tqdm import trange

from matex.agents import DQN


class Notice:
    _grey = "\x1b[37;40m"
    _blue = "\x1b[34;20m"
    _yellow = "\x1b[33;20m"
    _green = "\x1b[32;20m"
    _red = "\x1b[31;20m"
    _bold_red = "\x1b[31;1m"
    _reset = "\x1b[0m"

    def debug(self, *msg, show):
        msg = " ".join(msg)
        if show:
            print(f"{self._blue}[DEBUG] {msg}{self._reset}")

    def info(self, *msg):
        msg = " ".join(msg)
        print(f"{self._grey}[INFO] {msg}{self._reset}")

    def warning(self, *msg):
        msg = " ".join(msg)
        print(f"{self._yellow}[WARNING] {msg}{self._reset}")

    def error(self, *msg):
        msg = " ".join(msg)
        print(f"{self._red}[ERROR] {msg}{self._reset}")

    def critical(self, *msg):
        msg = " ".join(msg)
        print(f"{self._bold_red}[CRITICAL] {msg}{self._reset}")


def play_agent(env_name, ckpt_path, num_episodes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name, render_mode="human")
    agent = DQN.load(
        DQN,
        ckpt_path=ckpt_path,
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_size=64,
        device=device,
    )

    state, _ = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float).view(1, -1)
    terminated = False

    with trange(num_episodes) as pbar:
        for ep in pbar:
            pbar.set_description(f"[TEST] Episode: {ep+1:>5}")
            while not terminated:
                action = agent.act(state, deterministic=True)
                state, _, terminated, _, _ = env.step(action.item())
                state = torch.tensor(state, device=device, dtype=torch.float).view(1, -1)

    # movie = mpy.VideoFileClip("./runs/rl-video-episode-0.mp4")
    # movie.write_gif("./runs/rl-video-episode-0.gif")


def play_human():
    pass
