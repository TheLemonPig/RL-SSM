from typing import List
import numpy as np

from distribution import Distribution
from simple import SimpleRL


class MultiRL:

    def __init__(self, n_trials: int, n_participants: int, distributions: List[Distribution]):
        self.participants: List[SimpleRL] = [
            SimpleRL(n_trials, distributions) for _ in range(n_participants)
        ]
        self.alphas = list()
        self.temperatures = list()

    def simulate(self, alpha_a, alpha_b, temperature_a, temperature_b):
        group_data = []
        for idx, participant_model in enumerate(self.participants):
            # sample participant parameters
            alpha = np.random.beta(alpha_a, alpha_b)
            temperature = np.random.beta(temperature_a, temperature_b)
            self.alphas.append(alpha)
            self.temperatures.append(temperature)
            participant_rewards, participants_choices = np.vstack(participant_model.simulate(alpha, temperature))
            assert sum(np.isnan(participant_rewards)) == 0, participant_rewards
            trial_col = np.arange(participant_model.n_trials).reshape((-1,1))
            idx_col = np.ones_like(trial_col) * idx
            group_data.append(np.concatenate([idx_col,participant_rewards.reshape((-1,1)),participants_choices.reshape((-1,1)),trial_col],axis=1))
        return np.concatenate(group_data)

    def get_params(self):
        return self.alphas, self.temperatures


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    mean_rewards = [-0.3, 0.7]
    dists = [Distribution(np.random.normal, {"loc": mn, "scale": 1.0}) for mn in mean_rewards]
    num_trials = 1000
    n_trials = 50
    n_participants = 10
    n_choices = len(dists)
    multi_rl_model = MultiRL(n_trials=n_trials, n_participants=n_participants, distributions=dists)
    data = multi_rl_model.simulate(alpha_a=2.0, alpha_b=5.0, temperature_a=2.0, temperature_b=2.0)
    multi_qs = [participant.q_trace for participant in multi_rl_model.participants]
    print(multi_qs)
    # for i in range(len(multi_qs)):
    #     qs = multi_qs[i]
    #     plt.plot(qs[:, 1], c=(1 - i / n_participants, 0, 0, 1))
    #     plt.plot(qs[:, 0], c=(0, 1 - i / n_participants, 0, 1))
    # plt.show()
