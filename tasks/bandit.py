from typing import Union, List, Tuple
from functools import partial
import numpy as np

# TODO: Document class and move test script into a pytest file


class BanditTask:

    def __init__(self, dists: List[Tuple[Union[float, int], Union[float, int]]]):
        self.distributions = [partial(np.random.normal, **{"loc": dist[0], "scale": dist[1]}) for dist in dists]
        self.n_choices = len(self.distributions)

    def sample(self, i: int):
        if (isinstance(i, int) or isinstance(i, np.integer)) and 0 <= i < self.n_choices:
            return self.distributions[i]()
        else:
            raise KeyError(f'Invalid input: {i}\nInput should be integer in range 0-{self.n_choices-1}')


if __name__ == "__main__":
    # Parameters
    n_samples = 1000
    dists_params = [(-0.3, 0.1), (0.7, 0.1)]
    task = BanditTask(dists_params)

    # Test
    for idx in range(len(dists_params)):
        print(f"mean reward: {dists_params[idx][0]}")
        samples = []
        for _ in range(n_samples):
            samples.append(task.sample(idx))
        print(f"sample prediction: {sum(samples)/n_samples}")
