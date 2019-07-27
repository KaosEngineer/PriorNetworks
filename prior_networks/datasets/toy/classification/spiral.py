#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch
from torch.utils.data import Dataset


class SpiralDataset(Dataset):
    """ The Spiral Dataset """

    def __init__(self, n_examples_per_class, noise, n_spirals=3, seed=100):
        self.n_examples_per_class = n_examples_per_class
        self.size = n_examples_per_class * n_spirals
        self.noise = noise
        self.n_spirals = n_spirals
        self.seed = seed

        self.x, self.y = create_spirals(n_examples_per_class, n_spirals=n_spirals, noise=noise, seed=seed)
        return

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def plot(self, ax=None, s=10.0, alpha=0.7):
        """Plots the dataset"""
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(aspect='equal')
        colors = sns.cubehelix_palette(self.n_spirals, start=0.2, rot=-0.7, light=0.75)
        for i in range(self.n_spirals):
            plt.scatter(*np.hsplit(self.x[self.y == i], 2), color=colors[i], s=s, alpha=alpha)
        return ax

    def plot_with_rand_order(self, ax=None, s=10.0, alpha=0.5):
        """Plots the dataset, each point one by one. Since the points are drawn in random order,
        the effect where in regular plotting one class that was plotted after the other can completely
        hide the other one. Takes longer time, bur produces nicer plots"""
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(aspect='equal')
        colors = sns.color_palette('husl', self.n_spirals)

        shuffle_idx = np.random.permutation(self.x.shape[0])
        x_shuffled = self.x[shuffle_idx]
        y_shuffled = self.y[shuffle_idx]
        for i in range(x_shuffled.shape[0]):
            plt.scatter(x_shuffled[i, 0], x_shuffled[i, 1], color=colors[y_shuffled[i]], s=s, alpha=alpha)
        return ax

    def calc_prob_of_x(self, data_points, n_samples):
        """
        Calculate the true probability p(x) under this generative model using trapesoidal rule
        for the integral over the latent variable.
        :param data_points: an [N, 2] shaped array where N is the number of data points to calc.
        probability for
        :param n_samples: how many samples to use for the numerical integral
        :return: an [N] shaped numpy array
        """
        if data_points.ndim == 1:
            assert data_points.shape[0] == 2
            data_points.reshape([None, 2])
        else:
            assert data_points.ndim == 2
            assert data_points.shape[1] == 2

        prob_of_x_given_s = []

        for k in range(self.n_spirals):
            prob_of_x_given_s.append(self.calc_prob_of_x_given_class(data_points, spiral_class=k, n_samples=n_samples))
        prob_of_x_given_s = np.stack(prob_of_x_given_s, axis=1)
        return np.mean(prob_of_x_given_s, axis=1)

    def calc_prob_of_x_given_class(self, data_points, spiral_class, n_samples):
        angle_offset = spiral_class * 2 * np.pi / self.n_spirals  # The angle offset of spiral k
        # Integrate over uniform u for each of the spirals
        # to get p(x|s=k)
        prob_of_x_given_s_u = self.calc_prob_of_x_given_u(data_points, u=np.linspace(1e-30, 1., n_samples),
                                                          angle_offset=angle_offset)
        res = np.trapz(prob_of_x_given_s_u, dx=1. / n_samples, axis=1)
        return res

    def calc_prob_of_x_given_u(self, data_points, u, angle_offset):
        """Calculate p(x|u, s=k) where u is the uniformly distributed latent var and s is the spiral class"""
        x_comp_mean = -np.cos(6 * np.pi * np.sqrt(u) + angle_offset) * (6 * np.pi) ** 2 * u
        y_comp_mean = np.sin(6 * np.pi * np.sqrt(u) + angle_offset) * (6 * np.pi) ** 2 * u

        noise_std = (6 * np.pi) ** (3 / 2) * u ** (3 / 4) * self.noise

        # Expand above to match dimensions of data_points
        x_comp_mean, y_comp_mean, noise_std = map(lambda x: np.tile(x, (data_points.shape[0], 1)),
                                                  (x_comp_mean, y_comp_mean, noise_std))

        noise_x = (data_points[:, 0, None] - x_comp_mean) / noise_std
        noise_y = (data_points[:, 1, None] - y_comp_mean) / noise_std

        x_comp_prob = stats.norm.pdf(np.nan_to_num(noise_x))
        y_comp_prob = stats.norm.pdf(np.nan_to_num(noise_y))
        return x_comp_prob * y_comp_prob

    def optimal_predict(self, x, n_samples=1000):
        """Make the optimal prediction of model's class using Baye's rule and the generative model of the data."""
        prob_of_x_given_s = []
        for k in range(self.n_spirals):
            prob_of_x_given_s.append(self.calc_prob_of_x_given_class(x, spiral_class=k, n_samples=n_samples))
        prob_of_x_given_s = np.stack(prob_of_x_given_s, axis=1)
        probs = prob_of_x_given_s / prob_of_x_given_s.sum(axis=1, keepdims=True)
        return probs

    def optimal_predict_torch(self, x, n_samples=1000):
        probs = self.optimal_predict(x.cpu().numpy(), n_samples=n_samples)
        return torch.tensor(probs)

class OODSpiralDataset(Dataset):
    """ Genereate out of domain spiral dataset """
    def __init__(self, n_examples, noise=1.0, seed=100, radius=410):
        self.n_examples = n_examples
        self.size = self.n_examples
        self.seed = seed
        self.radius = radius

        self.x = create_circle_data(self.n_examples, radius=radius, noise=noise, seed=seed)
        return

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx]

    def plot(self, ax=None, s=10.0, alpha=0.001, color=(0.67, 0.41, 0.56)):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(aspect='equal')
        plt.scatter(*np.hsplit(self.x, 2), color=color, s=s, alpha=alpha)
        return ax

def create_single_spiral(n_points, angle_offset, noise=0.1):
    # Create numbers in the range [0., 6 pi], where the initial square root maps the uniformly
    # distributed points to lie mainly towards the upper limit of the range
    n = np.sqrt(np.random.rand(n_points, 1)) * 3 * (2 * np.pi)

    # Calculate the x and y coordinates of the spiral and add random noise to each coordinate
    x = -np.cos(n + angle_offset) * n ** 2 + np.random.randn(n_points, 1) * noise * n * np.sqrt(n)
    y = np.sin(n + angle_offset) * n ** 2 + np.random.randn(n_points, 1) * noise * n * np.sqrt(n)

    return np.hstack((x, y))

def create_spirals(n_points, n_spirals=3, noise=0.1, seed=100):
    """
    Returns the three spirals dataset.
    """
    np.random.seed(seed)

    angle_separation = 2 * np.pi / n_spirals  # The angle separation between each spiral

    X, Y = [], []
    for i in range(n_spirals):
        X.append(create_single_spiral(n_points, angle_offset=angle_separation * i, noise=noise))
        Y.append(np.ones(n_points) * i)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.long)

def create_circle_data(n_points, radius=410, noise=0.01, seed=100):
    np.random.seed(seed)

    samples = np.random.randn(n_points, 2)
    # Sample random points on a circle
    circle = radius * samples / (np.sqrt(np.sum(samples ** 2, axis=1, keepdims=True)))
    noisy_circle = circle + noise * np.random.randn(n_points, 2)
    return np.asarray(noisy_circle, dtype=np.float32)

