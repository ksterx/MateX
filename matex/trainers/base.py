import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from matex.agents import DQN


class Trainer:
    def __init__(self, config):

        self.config = config

        render_mode = "human" if config.render else None
        self.env = gym.make(config.env_name, render_mode=render_mode)
        self.writer = SummaryWriter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = DQN(
            lr=config.lr,
            gamma=config.gamma,
            memory_size=config.agents.memory_size,
            batch_size=config.batch_size,
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            hidden_size=config.hidden_size,
            device=self.device,
        )

    def train(self):
        with trange(self.config.n_episodes) as pbar:
            best_step = 0
            for ep in pbar:
                pbar.set_description(f"Episode: {ep:>5}")

                state, _ = self.env.reset()
                state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)

                for step in range(self.config.max_steps):
                    action = self.agent.act(
                        state,
                        eps=self.config.eps_max,
                        prog_rate=ep / self.config.n_episodes,
                        eps_decay=self.config.eps_decay,
                        eps_min=self.config.eps_min,
                    )
                    next_state, _, done, _, _ = self.env.step(action.item())
                    next_state = torch.tensor(
                        next_state, device=self.device, dtype=torch.float
                    ).view(1, -1)

                    if done:
                        if step >= self.config.max_steps - 5:
                            reward = torch.tensor(
                                [[1.0]], device=self.device, dtype=torch.float
                            ).view(1, 1)
                        else:
                            reward = torch.tensor(
                                [[-1.0]], device=self.device, dtype=torch.float
                            ).view(1, 1)
                    else:
                        reward = torch.tensor([[0.01]], device=self.device, dtype=torch.float).view(
                            1, 1
                        )

                    self.agent.memorize(state, action, next_state, reward, done)
                    self.agent.learn()

                    state = next_state

                    if step % self.config.agents.target_net_update_freq == 0:
                        self.agent.update_target_network(
                            soft_update=True, tau=self.config.agents.tau
                        )

                    if done:
                        self.writer.add_scalar("step", step + 1, ep)
                        if step + 1 >= best_step:
                            best_step = step + 1
                            self.agent.save("./runs/dqn.pth")
                        break

                pbar.set_postfix(step=f"{step+1:*>3}")

        self.writer.close()

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def play(self, n_episodes=1):
        env = gym.make(self.config.env_name, render_mode="human")
        state, _ = env.reset()
        state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)
        done = False

        for _ in range(n_episodes):
            action = self.agent.act(state, epsilon=0.0)
            state, _, done, _, _ = env.step(action.item())
            state = torch.tensor(state, device=self.device, dtype=torch.float).view(1, -1)

            if done:
                break
