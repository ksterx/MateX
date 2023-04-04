import torch


class DQN:
    def __init__(
        self, lr, gamma, memory_size, batch_size, state_size, action_size, hidden_size, device
    ):
        self.lr = lr
        self.gamma = gamma
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        self.device = device

        self.q_network = QNetwork(state_size, action_size, hidden_size, self.device).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size, self.device).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def act(self, state, epsilon):
        return self.q_network.act(state, epsilon)

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
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def memorize(self, state, action, next_state, reward, done):
        self.memory.add(state, action, next_state, reward, done)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
