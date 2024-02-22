import numpy as np
from typing import List


from rl.distribution import Distribution


def softmax(qs, tau):
    safe_tau = np.abs(tau) + 1e-2
    return np.exp(qs / safe_tau) / np.exp(qs / safe_tau).sum()


def rescola(qt, lr, reward):
    return qt + lr * (reward - qt)


class SimpleRL:

    def __init__(self, n_trials: int, distributions: List[Distribution]):
        self.n_choices: int = len(distributions)
        self.n_trials: int = n_trials
        self.distributions: List[Distribution] = distributions
        self.qs: np.array = np.ones((self.n_choices,)) * 0.  # check whether I should set to 0.5
        self.q_trace: np.array = np.ones((self.n_trials, self.n_choices))
        self.rewards: np.array = np.zeros(self.n_trials, dtype=np.int32)
        self.choices: np.array = np.zeros(self.n_trials, dtype=np.int32)

    def simulate(self, alpha, temperature):
        for i in range(self.n_trials):
            # Q-values are recorded to trace
            self.q_trace[i] = self.qs
            # softmax decision function
            ps = softmax(self.qs, temperature)
            # choice made based on weighted probabilities of Q-values
            choice = np.random.choice(a=self.n_choices, size=1, p=ps)[0]
            # choice is recorded to trace
            self.choices[i] = choice
            # reward calculated
            dist = self.distributions[choice]  # supply a list of distributions to choose from
            reward = dist()  # sample from distribution by calling it
            # Q-values updated
            self.rewards[i] = reward
            self.qs[choice] = rescola(self.qs[choice], alpha, reward)
        # Q-values trace returned
        # main data to be returned (basis for fits), is choices and rewards per trial
        return self.rewards, self.choices


if __name__ == "__main__":
    # Parameters
    seed = 0
    np.random.seed(seed)
    mean_rewards = [-0.3, 0.7]
    dists = [Distribution(np.random.normal, {"loc": mn, "scale": 1.0}) for mn in mean_rewards]
    num_trials = 1000
    temperature = 1
    alpha = 0.01

    # Simulate RL Model
    rl_model = SimpleRL(n_trials=num_trials, distributions=dists)
    rewards, choices = rl_model.simulate(alpha, temperature)
    qs_predict = rl_model.q_trace
    
    # Plot Q-values
    #plt.plot(qs_predict)
