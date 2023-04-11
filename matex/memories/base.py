import random

from matex import Experience


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.exps = []
        self.idx = 0

    def add(self, experience: Experience = None, **kwargs):
        if len(self.exps) < self.capacity:
            self.exps.append(None)
        if experience:
            self.exps[self.idx] = experience
        elif kwargs:
            try:
                self.exps[self.idx] = Experience(**kwargs)
            except TypeError:
                raise TypeError(f"kwargs must be {Experience.__annotations__}")
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.exps, batch_size)

    def __len__(self):
        return len(self.exps)
