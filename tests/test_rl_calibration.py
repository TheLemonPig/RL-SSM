# Packages
import numpy as np
import random
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
from tqdm import tqdm

# Local Code
from ..rl.rl_pymc import LogLike
from rl_ll import rl_ll
from distribution import Distribution
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


def rl_calibration():
    quantiles = np.zeros((nrep, 2))
    for rep_i in tqdm(range(nrep)):
        a_true = np.random.uniform(*a_range)
        b_true = np.random.uniform(*b_range)
        y_R, y_C = sample(trials, dists, a_true, b_true)
        data = fit_pymc(y_R, y_C, prior_tuple)
        quantiles[rep_i, 0] = np.mean(data.posterior.a > a_true)
        quantiles[rep_i, 1] = np.mean(data.posterior.b > b_true)
    return quantiles

def test_rl_calibration(seed):    
    np.random.seed(seed)
    random.seed(seed)
    
    # Calibration Parameters
    nrep = 10
    trials = 500
    a_range = (0.01, 0.99)
    b_range = (0.01, 0.99)
    mean_rewards = [-0.3, 0.7]
    dists = [Distribution(np.random.normal, {"loc": mn, "scale": 1.0}) for mn in mean_rewards]
    prior_dist = (pm.Uniform, pm.Uniform)
    args = (["a", 1e-4, 1e+0],
            ["b", 1e-4, 1e+0])
    prior_tuple = (prior_dist, args)

    quants = rl_calibration()
    quants.to_file('tests/quantiles.csv')
    print(quants)
    

if __name__ == "__main__":
    seed = 0
    test_rl_calibration(seed)
    