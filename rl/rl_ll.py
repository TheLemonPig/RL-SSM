from distribution import Distribution
from simple import softmax, SimpleRL
from rl_pytensor import pt_model
import numpy as np


# blackbox loglikelihood function for RL Model
def rl_ll(params, R, C):
    a, b = params
    m = np.max(C)
    q = np.zeros((m+1,))
    q_predict = pt_model()
    Q_predict = q_predict(C, R, q, a)
    probabilities = softmax(Q_predict, b)   # make sure I use temperature here
    selected_probabilities = probabilities[np.arange(C.shape[0]), np.int32(C)]
    log_likelihood = np.sum(np.log(selected_probabilities))
    return log_likelihood


def rl_nll(params, R, C):
    return -rl_ll(params, R, C)


if __name__ == "__main__":
    # Creating Ground Truth Data
    n_trials = 100
    mean_rewards = [-0.3, 0.7]
    dists = [Distribution(np.random.normal, {"loc": mn, "scale": 1.0}) for mn in mean_rewards]
    # Creating Ground Truth Choices
    a_true = 0.1
    b_true = 0.8
    rl_model = SimpleRL(n_trials=n_trials, distributions=dists)
    R_true, C_true = rl_model.simulate(a_true, b_true)
    ll = rl_ll((a_true, b_true), R_true, C_true)
    print(ll)
