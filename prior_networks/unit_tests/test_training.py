import context
import pytest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset

from prior_networks.training import Trainer
from prior_networks.priornet.training import TrainerWithOOD
from prior_networks.priornet.losses import PriorNetMixedLoss, \
    DirichletReverseKLLoss


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def new_model():
    net = ToyNet()
    return net


def make_dataset():
    input_data = np.random.randn(100, 3)
    targets = np.random.randint(0, 20, [100])
    input_data, targets = map(lambda data: torch.Tensor(data),
                              (input_data, targets))
    targets = targets.long()
    dataset = TensorDataset(input_data, targets)
    return dataset


@pytest.fixture
def new_trainer(new_model):
    train_dataset = make_dataset()
    test_dataset = make_dataset()
    trainer = Trainer(new_model, nn.CrossEntropyLoss(), train_dataset, test_dataset,
                      optim.SGD, optim.lr_scheduler.ExponentialLR,
                      optimizer_params={'lr': 1e-3},
                      scheduler_params={'gamma': 0.5},
                      batch_size=10)
    return trainer


def test_trainer_train(new_trainer):
    trainer: Trainer = new_trainer
    trainer.train(n_epochs=2)


def test_trainer_test(new_trainer):
    trainer: Trainer = new_trainer
    trainer.train(n_epochs=2)
    trainer.test()


@pytest.fixture
def new_trainer_with_ood(new_model):
    train_dataset = make_dataset()
    ood_dataset = make_dataset()
    test_dataset = make_dataset()

    criterion = PriorNetMixedLoss(
        [DirichletReverseKLLoss(target_concentration=1e3),
         DirichletReverseKLLoss()],
        [1., 2.])
    trainer = TrainerWithOOD(new_model, criterion,
                             train_dataset,
                             ood_dataset, test_dataset,
                             optim.SGD, optim.lr_scheduler.ExponentialLR,
                             optimizer_params={'lr': 1e-3},
                             scheduler_params={'gamma': 0.5},
                             batch_size=10)
    return trainer


def test_trainer_with_ood(new_trainer_with_ood):
    trainer: Trainer = new_trainer_with_ood
    trainer.train(n_epochs=2)
    trainer.test()

