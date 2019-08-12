from typing import Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F


class NormalWishartKLLoss:
    """
    Can be applied to any model which returns logits

    """

    def __init__(self, target_concentration=1e3, concentration=1.0, reverse=True):
        pass

    def __call__(self, logits, labels, mean=True):
        pass

    def forward(self, alphas, labels, mean):
        pass

    def compute_loss(self, alphas, labels: Optional[torch.tensor] = None):
        pass


def nwpn_kl_divergence(pmean, log_prec, log_mean_belief, log_prec_belief, epsilon=1e-10):
    return

def nwpn_rkl_divergence(pmean, log_prec, log_mean_belief, log_prec_belief, epsilon=1e-10):
    return
