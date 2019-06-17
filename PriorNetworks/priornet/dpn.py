import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln, digamma
from torch import nn

from PriorNetworks.util_pytorch import categorical_entropy_torch


# TODO Decide what is a static method and what is a function call...

class PriorNet(nn.Module):
    """Prior Net is a wrapper around a regular model that returns logits. It then allows for computations
    of various Prior Network related statistics."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def alphas(self, x):
        return torch.exp(self.forward(x))

    def mutual_information(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0

        expected_entropy = self.expected_entropy_from_alphas(alphas, alpha0)
        entropy_of_exp = categorical_entropy_torch(probs)
        mutual_info = entropy_of_exp - expected_entropy
        return mutual_info

    def entropy_of_expected(self, x):
        probs = F.softmax(self.model(x), dim=1)
        entropy_of_exp = categorical_entropy_torch(probs)
        return entropy_of_exp

    def expected_entropy(self, x):
        alphas = self.alphas(x)
        return self.expected_entropy_from_alphas(alphas)

    def diffenrential_entropy(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)

        return torch.sum(torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
                         dim=1) - torch.lgamma(alpha0)

    def epkl(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        return alphas.size()[1] / alpha0

    def confidence(self, x):
        return torch.max(F.softmax(self.forward(x)), dim=1)

    @staticmethod
    def expected_entropy_from_alphas(alphas, alpha0=None):
        if alpha0 is None:
            alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        expected_entropy = -torch.sum((alphas / alpha0) * (torch.digamma(alphas + 1) - torch.digamma(alpha0 + 1)),
                                      dim=1)
        return expected_entropy

    @staticmethod
    def uncertainty_metrics(logits):
        """Calculates mutual info, entropy of expected, and expected entropy, EPKL and Differential Entropy uncertainty metrics for
        the data x."""
        alphas = torch.exp(logits)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0

        epkl = (alphas.size()[1]-1.0)/alphas

        dentropy = torch.sum(torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
                         dim=1) - torch.lgamma(alpha0)

        conf = torch.max(probs,dim=1)

        expected_entropy = -torch.sum((alphas / alpha0) * (torch.digamma(alphas + 1) - torch.digamma(alpha0 + 1)),
                                      dim=1)
        entropy_of_exp = categorical_entropy_torch(probs)
        mutual_info = entropy_of_exp - expected_entropy
        return conf, entropy_of_exp, expected_entropy, mutual_info, epkl, dentropy


def dirichlet_prior_network_uncertainty(logits, epsilon=1e-10):
    """

    :param logits:
    :param epsilon:
    :return:
    """

    alphas = np.exp(logits)
    alpha0 = np.sum(alphas, axis=1, keepdims=True)
    probs = alphas / alpha0

    conf = np.max(probs, axis=1)

    entropy_of_exp = -np.sum(probs*np.log(probs+epsilon), axis=1)
    expected_entropy = -np.sum((alphas / alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1.0)), axis=1)
    mutual_info = entropy_of_exp - expected_entropy

    epkl = (alphas.size()[1] - 1.0) / alphas

    dentropy = np.sum(gammaln(alphas) - (alphas - 1.0) * (digamma(alphas) - digamma(alpha0)), axis=1) - gammaln(alpha0)

    uncertainty = {'confidence' : conf,
                   'entropy_of_expected' : entropy_of_exp,
                   'expected_entropy': expected_entropy,
                   'mutual_information': mutual_info,
                   'EPKL': epkl,
                   'differential_entropy': dentropy}

    return uncertainty
