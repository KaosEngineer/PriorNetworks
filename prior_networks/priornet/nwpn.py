import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln, digamma
from torch import nn

from prior_networks.util_pytorch import categorical_entropy_torch


# TODO Decide what is a static method and what is a function call...

class NormalWishartPriorNet(nn.Module):
    """Normal Wishart Prior Net is a wrapper around a regular model that returns logits. It then allows for computations
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

        return torch.sum(
            torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
            dim=1) - torch.lgamma(alpha0)

    def epkl(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        return alphas.size()[1] / alpha0

    @staticmethod
    def expected_entropy_from_alphas(alphas, alpha0=None):
        if alpha0 is None:
            alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        expected_entropy = -torch.sum(
            (alphas / alpha0) * (torch.digamma(alphas + 1) - torch.digamma(alpha0 + 1)),
            dim=1)
        return expected_entropy

    @staticmethod
    def uncertainty_metrics(logits):
        """Calculates mutual info, entropy of expected, and expected entropy, EPKL and Differential Entropy uncertainty metrics for
        the data x."""
        alphas = torch.exp(logits)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0

        epkl = (alphas.size()[1] - 1.0) / alphas

        dentropy = torch.sum(
            torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
            dim=1) - torch.lgamma(alpha0)

        conf = torch.max(probs, dim=1)

        expected_entropy = -torch.sum(
            (alphas / alpha0) * (torch.digamma(alphas + 1) - torch.digamma(alpha0 + 1)),
            dim=1)
        entropy_of_exp = categorical_entropy_torch(probs)
        mutual_info = entropy_of_exp - expected_entropy
        return conf, entropy_of_exp, expected_entropy, mutual_info, epkl, dentropy


def normal_wishart_prior_network_uncertainty(p_mean, p_precision, mean_belief, precision_belief, epsilon=1e-10):


    uncertainty = {'entropy_of_expected': entropy_of_exp,
                   'expected_entropy': expected_entropy,
                   'mutual_information': mutual_info,
                   'EPKL': epkl,
                   'differential_entropy': np.squeeze(dentropy)}

    return uncertainty
