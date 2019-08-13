import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln, digamma, multigammaln
from torch import nn

from prior_networks.util_pytorch import categorical_entropy_torch


# TODO Decide what is a static method and what is a function call...
class NormalWishartPriorNet(nn.Module):
    """Normal Wishart Prior Net is a wrapper around a regular model that returns logits. It then allows for computations
    of various Prior Network related statistics."""

    def __init__(self, n_in, n_out):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.pmean = nn.Linear(in_features=n_in, out_features=n_out)
        self.log_prec = nn.Linear(in_features=n_in, out_features=n_out)
        self.log_mean_belief = nn.Linear(in_features=n_in, out_features=1)
        self.log_prec_belief = nn.Linear(in_features=n_in, out_features=1)

    def forward(self, x):
        pmean = self.pmean(x)
        log_prec = self.log_prec(x)
        log_mean_belief = self.log_mean_belief(x)
        log_prec_belief = self.log_prec_belief(x)

        return pmean, log_prec, log_mean_belief, log_prec_belief

    def mutual_information(self, x):
        pass

    def entropy_of_expected(self, x):
        pass

    def expected_entropy(self, x):
        pass

    def diffenrential_entropy(self, x):
        pass

    def epkl(self, x):
        pass


class NormalInverseWishartPriorNet(nn.Module):
    """Normal Inverse Wishart Prior Net is a wrapper around a regular model that returns logits. It then allows for computations
    of various Prior Network related statistics."""

    def __init__(self, n_in, n_out):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out

        self.pmean = nn.Linear(in_features=n_in, out_features=n_out)
        self.log_pscatter = nn.Linear(in_features=n_in, out_features=n_out)
        self.log_mean_belief = nn.Linear(in_features=n_in, out_features=1)
        self.log_pscatter_belief = nn.Linear(in_features=n_in, out_features=1)

    def forward(self, x):
        pmean = self.pmean(x)
        log_pscatter = self.log_pscatter(x)
        pmean_belief = torch.exp(self.log_mean_belief(x))
        pscatter_belief = (torch.exp(self.log_pscatter_belief(x))+self.n_out-1.0)

        return pmean, log_pscatter, pmean_belief, pscatter_belief

    def mutual_information(self, x):
        pass

    def entropy_of_expected(self, x):
        pass

    def expected_entropy(self, x):
        pass

    def diffenrential_entropy(self, x):
        pass

    def epkl(self, x):
        pass


def entropy_of_expected(pmean, pscatter, pmean_belief, pscatter_belief, epsilon=1e-10):
    K = pscatter.shape[1]

    eoe = gammaln((pscatter_belief-K+1)/2.0) - gammaln((pscatter_belief+1)/2.0)- \
         K/2.0*np.log((pscatter_belief-K+1)*np.pi) - \
         (pscatter_belief+1)/2.0*(digamma((pscatter_belief+1)/2.0)-digamma((pscatter_belief-K+1)/2.0)) + \
        0.5*np.sum(np.log(pscatter), axis=1, keepdims=True) \
        + K/2.0*(np.log(pmean_belief+1)-np.log(pmean_belief*(pscatter_belief-K+1)))

    return eoe

def expected_entropy(pmean, pscatter, pmean_belief, pscatter_belief, epsilon=1e-10):
    K = pscatter.shape[1]
    const = K + K*np.log(np.pi)
    exe = const + np.sum(np.log(pscatter), axis=1, keepdims=True) \
          - np.sum(digamma((pscatter_belief-K+np.arange(1.0, K+1))/2.0), axis=1, keepdims=True)

    return 0.5*exe

def mutual_information(pmean, pscatter, pmean_belief, pscatter_belief, epsilon=1e-10):
    eoe = entropy_of_expected(pmean, pscatter, pmean_belief, pscatter_belief)
    exe = expected_entropy(pmean, pscatter, pmean_belief, pscatter_belief)
    return eoe-exe

def expected_pairwise_KL(pmean, pscatter, pmean_belief, pscatter_belief, epsilon=1e-10):
    K = pscatter.shape[1]
    epkl = (pscatter_belief*K)/(pscatter_belief-K-1)-K + ((pscatter_belief*K)/(pscatter_belief-K-1)+K)/pmean_belief
    return 0.5*epkl

def differential_entropy(pmean, pscatter, pmean_belief, pscatter_belief, epsilon=1e-10):
    K = pscatter.shape[1]
    de = multigammaln(pscatter_belief/2.0, K) + \
         (pscatter_belief+1)*K/2.0*(np.sum(np.log(pscatter), keepdims=True, axis=1)-K*np.log(2.0)) \
        - (pscatter_belief+K+2)/2.0*np.sum(digamma((pscatter_belief-K+np.arange(1.0, K+1))/2.0),
                                           axis=1, keepdims=True) \
        + K*np.log(np.pi/pmean_belief)

    return de

def niwpn_uncertainty(pmean, pscatter, pmean_belief, pscatter_belief, epsilon=1e-10):
    eoe = entropy_of_expected(pmean, pscatter, pmean_belief, pscatter_belief)
    exe = expected_entropy(pmean, pscatter, pmean_belief, pscatter_belief)
mi = eoe-exe
    epkl = expected_pairwise_KL(pmean, pscatter, pmean_belief, pscatter_belief)
    de = differential_entropy(pmean, pscatter, pmean_belief, pscatter_belief)

    uncertainty = {'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mi,
                   'EPKL': epkl,
                   'differential_entropy': de}

    return uncertainty
