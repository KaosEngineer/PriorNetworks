from torch.utils.data import Dataset

import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


class MixtureOfGaussiansDataset(Dataset):
    """ The Toy Three Class Dataset """

    def __init__(self, size, noise, scale, OOD=False, seed=100):
        self.scale = scale
        self.size = size
        self.noise = noise
        self.seed = seed
        self.OOD = OOD

        self.x, self.y, self.ood_data = create_mixture_of_gaussians(size=self.size,
                                                               noise=self.noise,
                                                               scale=self.scale,
                                                               seed=self.seed)
        return

    def __len__(self):
        return self.size * 3

    def __getitem__(self, idx):
        if self.OOD:
            return self.ood_data[idx], np.long(-1)
        else:
            return self.x[idx], self.y[idx]

    def plot(self, ax=None, s=10.0, alpha=0.7):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(aspect='equal')
        if self.OOD:
            plt.scatter(*np.hsplit(self.OOD, 2), s=s, alpha=alpha)
        else:
            colors = sns.cubehelix_palette(3, start=0.2, rot=-0.7, light=0.75)
            for i in range(3):
                plt.scatter(*np.hsplit(self.x[self.y == i], 2), color=colors[i], s=s, alpha=alpha)

        return ax


def create_mixture_of_gaussians(size, noise, scale=4.0, seed=100):
    """

    :param size:
    :param noise:
    :param scale:
    :param seed:
    :return:
    """
    init_means = scale * np.asarray([[0.0, 1.0],
                                     [-np.sqrt(3) / 2, -1.0 / 2],
                                     [np.sqrt(3) / 2, -1.0 / 2]])
    init_weights = [1 / 3.0, 1 / 3.0, 1 / 3.0]
    np.random.seed(seed)

    gaussian1 = np.random.randn(size, 2) * noise + init_means[0]
    gaussian2 = np.random.randn(size, 2) * noise + init_means[1]
    gaussian3 = np.random.randn(size, 2) * noise + init_means[2]
    data = np.concatenate((gaussian1, gaussian2, gaussian3), axis=0)

    gmm = GMM(n_components=3,
              covariance_type='spherical',
              means_init=init_means,
              weights_init=init_weights,
              precisions_init=np.ones([3]) / noise,
              max_iter=1,
              random_state=1)
    gmm.fit(data)

    labels = np.concatenate((np.ones(size) * 0,
                             np.ones(size) * 1,
                             np.ones(size) * 2), axis=0)

    thresh = (norm.pdf(3.1)) ** 2
    OOD_data = np.random.uniform(size=[1000000, 2], low=-20.0, high=20.0)
    OOD_probs = np.exp(gmm.score_samples(OOD_data))
    inds = (OOD_probs < thresh) & (OOD_probs > thresh / 100000.0)
    OOD_data = OOD_data[inds]
    OOD_data = OOD_data[:3 * size]

    return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.long), np.asarray(OOD_data, dtype=np.float32)
