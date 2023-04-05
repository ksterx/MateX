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
        self, lr, gamma, memory_size, batch_size, state_size, action_size, hidden_size, device
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

    def act(self, state, epsilon, step=0, epsilon_decay=1.0, epsilon_min=0.01):
        if random.random() > max(epsilon * epsilon_decay * step, epsilon_min)):
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

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


def train():
    import gymnasium as gym
    import torch
    from torch.utils.tensorboard import SummaryWriter

    env = gym.make("CartPole-v1", render_mode="human")
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR = 1e-4
    GAMMA = 0.99
    MEMORY_SIZE = 1000
    BATCH_SIZE = 32
    N_EPISODES = 1000
    MAX_STEPS = 100
    EPSILON = 0.1
    EPSILON_DECAY = 0.99
    HIDDEN_SIZE = 64
    UPDATEW_FREQ = 10

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
            done = False

            for step in range(MAX_STEPS):
                action = agent.act(state, epsilon=EPSILON)
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

                if step % UPDATEW_FREQ == 0:
                    agent.update_target_network()

                if done:
                    writer.add_scalar("step", step, ep)
                    break

            pbar.set_postfix(step=f"{step:*>3}")


if __name__ == "__main__":
    train()
