import pymc as pm
import pytensor.tensor as pt
import numpy as np
from rl_ll import rl_ll
import matplotlib.pyplot as plt
import arviz as az
from simple import SimpleRL
from distribution import Distribution


# define a pytensor Op for our likelihood function
class LogLike(pt.Op):

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, r, c):
        # add inputs as class attributes
        self.likelihood = loglike
        self.R = r
        self.C = c

    def perform(self, node, inputs, outputs):
        (theta,) = inputs  # this will contain my variables
        outputs[0][0] = np.array(self.likelihood(theta, self.R, self.C))


def pymc_model(a_true, b_true, R_true, C_true):

    # create our Op
    logl = LogLike(rl_ll, R_true, C_true)

    # use PyMC to sampler from log-likelihood
    with pm.Model():
        # uniform priors on m and c
        # a = pm.Uniform("a", lower=1e-4, upper=1e-0)  # change priors to beta
        # b = pm.Uniform("b", lower=1e-4, upper=1e+0)  # alpha=1, beta=1; alpha=10 beta=10
        a = pm.Beta(name="a", alpha=1, beta=1)
        b = pm.Beta(name="b", alpha=1, beta=1)
        # a = pm.Beta(alpha=10,beta=10)
        # b = pm.Beta(alpha=10,beta=10)

        # convert m and c to a tensor vector
        theta = pt.as_tensor_variable([a, b])

        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))

        # Use custom number of draws to replace the HMC based defaults
        idata_mh = pm.sample(3000, tune=1000)

        # plot the traces
        az.plot_trace(idata_mh, lines=[("m", {}, a_true), ("c", {}, b_true)]);
        plt.show()
        az.plot_pair(idata_mh, kind="kde")
        plt.show()


if __name__ == "__main__":
    # Creating Ground Truth Data
    n_trials = 100
    mean_rewards = [-0.3, 0.7]
    dists = [Distribution(np.random.normal, {"loc": mn, "scale": 1.0}) for mn in mean_rewards]

    # Creating Ground Truth Choices
    a = 0.1
    b = 0.8
    rl_model = SimpleRL(n_trials=n_trials, distributions=dists)
    R, C = rl_model.simulate(a, b)

    # Run Model
    pymc_model(a, b, R, C)