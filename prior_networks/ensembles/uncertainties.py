import numpy as np

""" Numpy Implementation of Uncertainty Measures """

def kl_divergence(probs1, probs2, epsilon=1e-10):
    return np.sum(probs1 * (np.log(probs1 + epsilon) - np.log(probs2 + epsilon)), axis=1)


def expected_pairwise_kl_divergence(probs, epsilon=1e-10):
    kl = 0.0
    for i in range(probs.shape[1]):
        for j in range(probs.shape[1]):
            kl += kl_divergence(probs[:, i, :], probs[:, j, :], epsilon)
    return kl


def entropy_of_expected(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=1)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


def expected_entropy(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=1)


def mutual_information(probs, epsilon):
    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    return eoe - exe


def ensemble_uncertainties(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=1)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    mutual_info = eoe - exe

    epkl = expected_pairwise_kl_divergence(probs, epsilon)

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'EPKL': epkl}

    return uncertainty

""" Pytorch Implementation of Uncertainty Measures """
#TODO Pytorch Implementation of Uncertainty Measures