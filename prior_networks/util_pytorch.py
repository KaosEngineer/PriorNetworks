import os
import re
from pathlib import Path
from typing import Union
# import context.py
import numpy as np
import torch
import random

from torch import optim

from prior_networks.datasets import image

# TODO Add LeNet for MNIST and MNIST-like stuff

DATASET_DICT = {'MNIST': image.MNIST,
                'KMNIST': image.KMNIST,
                'FMNIST': image.FashionMNIST,
                'EMNIST': image.EMNIST,
                'SVHN': image.SVHN,
                'CIFAR10': image.CIFAR10,
                'CIFAR100': image.CIFAR100,
                'LSUN': image.LSUN,
                'TIM': image.TinyImageNet,
                'TIM-OOD': image.TinyImageNetConverse,
                'TIM-OOD-S1': image.TinyImageNetConverseS1,
                'TIM-OOD-S2': image.TinyImageNetConverseS2,
                'TIM-OOD-S3': image.TinyImageNetConverseS3,
                'TIM-OOD-S4': image.TinyImageNetConverseS4,
                'ImageNet': image.ImageNet}


def categorical_entropy(probs, axis=1, keepdims=False):
    """

    :param probs:
    :param axis:
    :param keepdims:
    :return:
    """
    return -np.sum(probs * np.log(probs, out=np.zeros_like(probs), where=(probs != 0.)), axis=axis,
                   keepdims=keepdims)


def categorical_entropy_torch(probs, dim=1, keepdim=False):
    """Calculate categorical entropy purely in torch"""
    log_probs = torch.log(probs)
    log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    entropy = -torch.sum(probs * log_probs, dim=dim, keepdim=keepdim)
    return entropy


def get_grid(xrange=(-500, 500), yrange=(-500, 500), resolution=200, dtype=np.float32):
    x = np.linspace(*xrange, resolution, dtype=dtype)
    y = np.linspace(*yrange, resolution, dtype=dtype)
    xx, yy = np.meshgrid(x, y, sparse=False)
    return xx, yy


def get_grid_eval_points(xrange, yrange, res):
    xx, yy = get_grid(xrange, yrange, res, dtype=np.float32)
    eval_points = torch.from_numpy(np.stack((xx.ravel(), yy.ravel()), axis=1))
    return eval_points


def select_device(device_name):
    if device_name is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device_name = device_name.strip()
        if re.search("^cuda:[0-9]$", device_name):
            assert torch.cuda.is_available()
            # Assert that the device selected isn't out of range:
            assert torch.cuda.device_count() > int(device_name[-1])
            device = torch.device(device_name)
            print(f"Using cuda device: {torch.cuda.get_device_name(device)} | {device_name}")
        elif device_name != "cpu":
            raise AttributeError(f"No such device allowed: {device_name}")
        device = torch.device(device_name)
    return device


def select_gpu(gpu_id: list):
    if torch.cuda.is_available() and len(gpu_id) > 0:
        assert torch.cuda.device_count() > max(gpu_id)
        device = torch.device(f"cuda:{gpu_id[0]}")
        print(f"Using device: {torch.cuda.get_device_name(device)} unit {gpu_id[0]}.")
    else:
        print(f"Using CPU device.")
        device = torch.device("cpu")

    return device


def set_random_seeds(seed: int) -> None:
    """Sets random seeds that could be used by PyTorch to a single value given by seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#

#
# def test_accuracy(predict_func, dataset, batch_size=100):
#     n_correct = 0  # Track the number of correct classifications
#     testloader = DataLoader(dataset, batch_size=batch_size,
#                             shuffle=False, num_workers=1)
#
#     with torch.no_grad():
#         for i, data in enumerate(testloader, 0):
#             inputs, labels = data
#             probs = predict_func(inputs)
#             n_correct += torch.sum(torch.argmax(probs, dim=1) == labels).item()
#     accuracy = n_correct / len(testloader.dataset)
#
#     return accuracy
#
#
# def test_error_rate(predict_func, dataset, batch_size=100):
#     return 1. - test_accuracy(predict_func, dataset, batch_size=batch_size)
#
#
# def test_nll(log_predict_func, dataset, batch_size=100):
#     loss_fun = torch.nn.NLLLoss(reduction='sum')
#     tot_loss = 0
#     testloader = DataLoader(dataset, batch_size=batch_size,
#                             shuffle=False, num_workers=1)
#     with torch.no_grad():
#         for i, data in enumerate(testloader, 0):
#             inputs, labels = data
#             probs = log_predict_func(inputs)
#             tot_loss += loss_fun(probs, labels).item()
#
#     mean_nll = tot_loss / len(testloader.dataset)
#     return mean_nll
#
##
#
# def cartestian_to_barometric(coord):
#     """Transform a set of cartesian coordinates to barometric. Assumes last dimension represents (x, y)
#     coordinates"""
#     corners = (np.array([0, 0]), np.array([1, 0]), np.array([0.5, 0.75 ** 0.5]))
#     barom = np.stack((np.linalg.norm(coord - corners[0], axis=1),
#                       np.linalg.norm(coord - corners[1], axis=1),
#                       np.linalg.norm(coord - corners[2], axis=1)), axis=1)
#     return barom


class TargetTransform:
    def __init__(self, target_concentration, gamma, ood=False):
        self.target_concentration = target_concentration
        self.gamma = gamma
        self.ood = ood

    def __call__(self, label):
        return self.forward(label)

    def forward(self, label):
        if self.ood:
            return (0, self.target_concentration, self.gamma)
        else:
            return (label, self.target_concentration, self.gamma)


def choose_optimizer(optimizer: str, learning_rate: float, weight_decay: float):
    if optimizer == 'SGD':
        optimizer = optim.SGD
        optimizer_params = {'lr': learning_rate, 'momentum': 0.9,
                            'nesterov': True,
                            'weight_decay': weight_decay}

    elif optimizer == 'ADAM':
        optimizer = optim.AdamW
        optimizer_params = {'lr': learning_rate, 'weight_decay': weight_decay}
    else:
        raise NotImplementedError

    return optimizer, optimizer_params
