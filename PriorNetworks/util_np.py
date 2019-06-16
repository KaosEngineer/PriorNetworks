import numpy as np
from scipy.special import gammaln, digamma

def dirichlet_prior_network_uncertainty(logits, epsilon=1e-10):
    """

    :param logits:
    :param epsilon:
    :return:
    """

    alphas = np.exp(logits)
    alpha0 = np.sum(alphas, dim=1, keepdim=True)
    probs = alphas / alpha0

    conf = np.max(probs, axism=1)

    entropy_of_exp = -np.sum(probs*np.log(probs+epsilon), axis=1)
    expected_entropy = -np.sum((alphas / alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1.0)), axis=1)
    mutual_info = entropy_of_exp - expected_entropy

    epkl = (alphas.size()[1] - 1.0) / alphas

    dentropy = np.sum(gammaln(alphas) - (alphas - 1.0) * (digamma(alphas) - digamma(alpha0)), axis=1) - gammaln(alpha0)

    uncertainty = {'confidence' : conf,
                   'entropy_of_expected' : entropy_of_exp,
                   'expected_entropy' : expected_entropy,
                   'mutual_information' : mutual_info,
                   'EPKL' : epkl,
                   'differential_entropy' : dentropy
    }

    return uncertainty

