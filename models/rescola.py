import numpy as np
from scipy.special import softmax

from tasks.bandit import BanditTask

# TODO: Document class and move test script into a pytest file


class Rescola:

    def __init__(self, task: BanditTask, n_trials: int):
        self.task = task
        self.n_choices: int = task.n_choices
        self.n_trials: int = n_trials
        self.qs: np.array = np.ones((self.n_choices, )) * 0.5
        self.q_trace: np.array = np.zeros((self.n_trials, self.n_choices))
        self.rewards: np.array = np.zeros((self.n_trials,))
        self.choices: np.array = np.zeros((self.n_trials,), dtype=np.int32)

    def simulate(self, alpha, beta):
        for i in range(self.n_trials):
            # Q-values are recorded to trace
            self.q_trace[i] = self.qs
            # softmax decision function
            ps = softmax(self.qs*beta)
            # choice made based on weighted probabilities of Q-values
            choice = np.random.choice(a=self.n_choices, size=1, p=ps)[0]
            # choice is recorded to trace
            self.choices[i] = choice
            # reward calculated
            reward = self.task.sample(choice)  # sample from task
            # Q-values updated
            self.rewards[i] = reward
            self.qs[choice] = self.qs[choice] + alpha * (reward - self.qs[choice])
        # Q-values trace returned
        # main data to be returned (basis for fits), is choices and rewards per trial
        return self.choices, self.rewards


if __name__ == "__main__":
    # Make Task
    dists_params = [(-0.3, 0.1), (0.7, 0.1)]
    task_ = BanditTask(dists_params)

    # Make Model
    n_samples = 1000
    model = Rescola(task_, n_samples)

    # Simulate
    alpha = 0.1
    beta = 1.0
    choices, rewards = model.simulate(alpha, beta)
    print(np.mean(choices), np.mean(rewards))