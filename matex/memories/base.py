import random
from collections import namedtuple

Experience = namedtuple("Experience", ("state", "action", "next_state", "reward", "terminated"))


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.exps = []
        self.idx = 0

    def add(self, state, action, next_state, reward, terminated):
        if len(self.exps) < self.capacity:
            self.exps.append(None)

        self.exps[self.idx] = Experience(state, action, next_state, reward, terminated)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.exps, batch_size)

    def __len__(self):
        return len(self.exps)
