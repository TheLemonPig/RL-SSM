from rl_pymc import LogLike
import pymc as pm
import pytensor.tensor as pt
from rl_ll import rl_ll
import numpy as np
import matplotlib.pyplot as plt
from distribution import Distribution
from tqdm import tqdm
from simple import SimpleRL


def fit_pymc(R, C, priors):
    # create our Op
    logl = LogLike(rl_ll, R, C)
    # use PyMC to sampler from log-likelihood
    with pm.Model():
        # uniform priors on m and c
        prior_dists, prior_args = priors
        a = prior_dists[0](*prior_args[0])
        b = prior_dists[1](*prior_args[1])
        # convert m and c to a tensor vector
        theta = pt.as_tensor_variable([a, b])
        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))
        # Use custom number of draws to replace the HMC based defaults
        idata_mh = pm.sample(2000, tune=1000)
    return idata_mh


# Calibrator Functions
def sample(n_trials, distributions, a, b):
    return SimpleRL(n_trials=n_trials, distributions=distributions).simulate(alpha=a, temperature=b)


def test_rl_calibration():
    quantiles = np.zeros((nrep, 2))
    for rep_i in tqdm(range(nrep)):
        a_true = np.random.uniform(*a_range)
        b_true = np.random.uniform(*b_range)
        y_R, y_C = sample(trials, dists, a_true, b_true)
        data = fit_pymc(y_R, y_C, prior_tuple)
        quantiles[rep_i, 0] = np.mean(data.posterior.a > a_true)
        quantiles[rep_i, 1] = np.mean(data.posterior.b > b_true)
    return quantiles


if __name__ == "__main__":
    # Calibration Parameters
    nrep = 10
    trials = 500
    a_range = (0.01, 0.99)
    b_range = (0.01, 0.99)
    mean_rewards = [-0.3, 0.7]
    dists = [Distribution(np.random.normal, {"loc": mn, "scale": 1.0}) for mn in mean_rewards]
    prior_dist = (pm.Uniform, pm.Uniform)
    args = (["a", 1e-4, 1e-0],
            ["b", 1e-4, 1e+0])  # alpha=1, beta=1; alpha=10 beta=10
    prior_tuple = (prior_dist, args)

    quants = test_rl_calibration()
    plt.hist(quants[:, 0])
    plt.show()
    plt.hist(quants[:, 1])
    plt.show()
    print(quants)
