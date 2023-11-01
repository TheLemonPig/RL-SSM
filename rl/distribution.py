from typing import Callable
from functools import partial
import numpy as np


class Distribution:

    def __init__(self, func: Callable, kwargs):
        self.func: Callable = partial(func, **kwargs)

    def __call__(self):
        return self.func.__call__()


if __name__ == "__main__":
    # Parameters
    n_samples = 1000
    mean_rewards = [-0.3, 0.7]
    dists = [Distribution(np.random.normal, {"loc": mn, "scale": 0.1}) for mn in mean_rewards]

    # Test
    for i in range(len(mean_rewards)):
        print(f"mean reward: {mean_rewards[i]}")
        dist = dists[i]
        samples = []
        for j in range(n_samples):
            samples.append(dist())
        print(f"sample prediction: {sum(samples)/n_samples}")

