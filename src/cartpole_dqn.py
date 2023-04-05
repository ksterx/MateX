import random
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

Experience = namedtuple("Experience", ("state", "action", "next_state", "reward", "done"))


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.exps = []
        self.idx = 0

    def add(self, state, action, next_state, reward, done):
        if len(self.exps) < self.capacity:
            self.exps.append(None)

        self.exps[self.idx] = Experience(state, action, next_state, reward, done)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.exps, batch_size)

    def __len__(self):
        return len(self.exps)


class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, device):
        super().__init__()

        self.action_size = action_size
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        print(self.net)

    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(
        self,
        lr,
        gamma,
        memory_size,
        batch_size,
        state_size,
        action_size,
        hidden_size,
        device,
    ):
        self.lr = lr
        self.gamma = gamma
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        self.device = device
        self.action_size = action_size

        self.q_network = QNet(state_size, action_size, hidden_size, self.device).to(self.device)
        self.target_network = QNet(state_size, action_size, hidden_size, self.device).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def act(
        self,
        state,
        epsilon=0.1,
        prog_rate=0,
        epsilon_decay=1.0,
        epsilon_min=0.01,
        deterministic=False,
    ):
        if deterministic:
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).view(1, 1)
        else:
            if random.random() > max(epsilon * epsilon_decay * (1 - prog_rate), epsilon_min):
                with torch.no_grad():
                    return torch.argmax(self.q_network(state)).view(1, 1)
            else:
                return torch.tensor(
                    [[random.randrange(self.action_size)]], device=self.device, dtype=torch.long
                )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        exps = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*exps))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        next_states = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward)
        dones = torch.tensor(batch.done, device=self.device, dtype=torch.float).view(-1, 1)

        self.q_network.eval()
        estimated_Qs = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_Qs = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_Qs = rewards + self.gamma * next_Qs * (1 - dones)

        self.q_network.train()
        loss = F.smooth_l1_loss(estimated_Qs, target_Qs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))

    def memorize(self, state, action, next_state, reward, done):
        self.memory.add(state, action, next_state, reward, done)

    def update_target_network(self, soft_update=False, tau=1.0):
        if soft_update:
            for target_param, q_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())


def train():
    import gymnasium as gym
    import torch
    from torch.utils.tensorboard import SummaryWriter

    env = gym.make("CartPole-v1", render_mode=None)
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR = 1e-4
    GAMMA = 0.99
    MEMORY_SIZE = 10000
    BATCH_SIZE = 128
    N_EPISODES = 1000
    MAX_STEPS = 200
    EPSILON = 0.1
    EPSILON_DECAY = 0.99
    EPSILON_MIN = 0.01
    HIDDEN_SIZE = 64
    TARGET_NET_UPDATE_FREQ = 10
    TAU = 0.01
    BEST_STEP = 0

    agent = DQN(
        lr=LR,
        gamma=GAMMA,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_size=HIDDEN_SIZE,
        device=device,
    )

    with trange(N_EPISODES) as pbar:
        for ep in pbar:
            pbar.set_description(f"Episode: {ep:>5}")

            state, _ = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float).view(1, -1)

            for step in range(MAX_STEPS):
                action = agent.act(
                    state,
                    epsilon=EPSILON,
                    prog_rate=ep / N_EPISODES,
                    epsilon_decay=EPSILON_DECAY,
                    epsilon_min=EPSILON_MIN,
                )
                next_state, _, done, _, _ = env.step(action.item())
                next_state = torch.tensor(next_state, device=device, dtype=torch.float).view(1, -1)

                if done:
                    if step >= MAX_STEPS - 5:
                        reward = torch.tensor([[1.0]], device=device, dtype=torch.float).view(1, 1)
                    else:
                        reward = torch.tensor([[-1.0]], device=device, dtype=torch.float).view(1, 1)
                else:
                    reward = torch.tensor([[0.01]], device=device, dtype=torch.float).view(1, 1)

                agent.memorize(state, action, next_state, reward, done)
                agent.learn()

                state = next_state

                if step % TARGET_NET_UPDATE_FREQ == 0:
                    agent.update_target_network(soft_update=True, tau=TAU)

                if done:
                    writer.add_scalar("step", step + 1, ep)
                    if step + 1 >= BEST_STEP:
                        BEST_STEP = step + 1
                        agent.save("./runs/dqn.pth")
                    break

            pbar.set_postfix(step=f"{step+1:*>3}")

    writer.close()

    def play():
        import gym
        import moviepy.editor as mpy

        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, "./runs/")
        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float).view(1, -1)
        done = False

        while not done:
            action = agent.act(state, deterministic=True)
            state, _, done, _, _ = env.step(action.item())
            state = torch.tensor(state, device=device, dtype=torch.float).view(1, -1)

        movie = mpy.VideoFileClip("./runs/rl-video-episode-0.mp4")
        movie.write_gif("./runs/rl-video-episode-0.gif")

    play()


if __name__ == "__main__":
    train()
