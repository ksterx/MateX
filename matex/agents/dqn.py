import random

import ray
import torch
from torch.nn import functional as F

from matex.memories import Experience, Memory
from matex.networks import QNet


@ray.remote
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
        is_ddqn=True,
    ):
        self.lr = lr
        self.gamma = gamma
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        self.device = device
        self.action_size = action_size
        self.is_ddqn = is_ddqn

        self.q_network = QNet(state_size, action_size, hidden_size, self.device).to(self.device)
        self.target_network = QNet(state_size, action_size, hidden_size, self.device).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def act(
        self,
        state,
        eps=0.1,
        prog_rate=0,
        eps_decay=1.0,
        eps_min=0.01,
        deterministic=False,
    ):
        if deterministic:
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).view(1, 1)
        else:
            if random.random() > max(eps * eps_decay * (1 - prog_rate), eps_min):
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
        terminateds = torch.tensor(batch.terminated, device=self.device, dtype=torch.float).view(
            -1, 1
        )

        self.q_network.eval()
        estimated_Qs = self.q_network(states).gather(1, actions)

        if self.is_ddqn:
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            next_Qs = self.target_network(next_states).gather(1, next_actions)

        else:
            next_Qs = self.target_network(next_states).max(1)[0].unsqueeze(1)

        target_Qs = rewards + self.gamma * next_Qs * (1 - terminateds)

        self.q_network.train()
        loss = F.smooth_l1_loss(estimated_Qs, target_Qs)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

    def save(self, ckpt_path):
        torch.save(self.q_network.state_dict(), ckpt_path)

    def load(self, ckpt_path):
        self.q_network.load_state_dict(torch.load(ckpt_path))

    def memorize(self, state, action, next_state, reward, terminated):
        self.memory.add(state, action, next_state, reward, terminated)

    def _update_target_network(self, soft_update=False, tau=1.0):
        if soft_update:
            for target_param, q_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def on_step_end(self, step, soft_update=False, tau=1.0, update_freq=10, **kwargs):
        if step % update_freq == 0:
            self._update_target_network(soft_update, tau)

    def on_episode_end(self):
        pass
