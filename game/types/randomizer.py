import random


class Randomizer:
    _base: random.Random

    def __init__(self, base: random.Random, seed: int):
        self._base = base
        self._seed = seed

    def randrange(self, n: int) -> int:
        return self._base.randrange(n)

    def choice(self, seq):
        return self._base.choice(seq)

    def shuffle(self, x) -> None:
        self._base.shuffle(x)

    def random(self) -> float:
        return self._base.random()

    def sample(self, population, k):
        return self._base.sample(population, k)
