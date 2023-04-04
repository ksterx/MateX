import random

from dqn import Memory


def test_memory():
    memory = Memory(10)
    memory.add(1, 2, 3, 4)
    memory.add(5, 6, 7, 8)
    memory.add(9, 10, 11, 12)
    print(memory.sample(2))
