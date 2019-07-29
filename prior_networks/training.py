import math
import os
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(self, model, criterion,
                 train_dataset, test_dataset,
                 optimizer,
                 scheduler=None,
                 optimizer_params: Dict[str, Any] = None,
                 scheduler_params: Dict[str, Any] = None,
                 batch_size=50,
                 device=None,
                 log_interval: int = 100,
                 test_criterion=None,
                 num_workers=4,
                 pin_memory=False,
                 checkpoint_path='./',
                 checkpoint_steps=0):
        assert isinstance(model, nn.Module)
        assert isinstance(train_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)

        self.model = model
        self.criterion = criterion
        self.device = device
        self.log_interval = log_interval
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.checkpoint_path = checkpoint_path
        self.checkpoint_steps = checkpoint_steps
        self.batch_size = batch_size
        if test_criterion is not None:
            self.test_criterion = test_criterion
        else:
            self.test_criterion = nn.CrossEntropyLoss()

        # Instantiate the optimizer
        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)

        # Instantiate the scheduler
        if scheduler_params is None:
            scheduler_params = {}
        self.scheduler = scheduler(self.optimizer, **scheduler_params)

        self.trainloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory)
        self.testloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory)

        # Lists for storing training metrics
        self.train_loss, self.train_accuracy, self.train_eval_steps = [], [], []
        # Lists for storing test metrics
        self.test_loss, self.test_accuracy, self.test_eval_steps = [], [], []
        self.steps: int = 0

    def _save_checkpoint(self, save_at_steps=False):
        if save_at_steps:
            checkpoint_name = 'checkpoint-' + str(self.steps) + '.tar'
        else:
            checkpoint_name = 'checkpoint.tar'

        torch.save({
            'steps': self.steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss
        }, os.path.join(self.checkpoint_path, checkpoint_name))

    def load_checkpoint(self, checkpoint_path, load_opt_state=False, load_scheduler_state=False):
        checkpoint = torch.load(checkpoint_path)
        self.steps = checkpoint['steps']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_loss = checkpoint['train_loss']
        self.test_loss = checkpoint['test_loss']

        if load_opt_state:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_scheduler_state:
            self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    def train(self, n_epochs=None, n_iter=None, resume=False):
        # Calc num of epochs
        init_epoch = 0
        if n_epochs is None:
            assert isinstance(n_iter, int)
            n_epochs = math.ceil(n_iter / len(self.trainloader))
        else:
            assert isinstance(n_epochs, int)

        if resume:
            init_epoch = math.floor(self.steps/math.ceil(len(self.trainloader)/self.batch_size))
            print(init_epoch)

        for epoch in range(init_epoch, n_epochs):
            print(f'Training epoch: {epoch + 1} / {n_epochs}')
            # Train
            start = time.time()
            self.scheduler.step()
            self._train_single_epoch()
            self._save_checkpoint()
            # Test
            self.test(time=time.time() - start)
        return

    def _train_single_epoch(self):

        # Set model in train mode
        self.model.train()

        for i, data in enumerate(self.trainloader, 0):
            # Get inputs
            inputs, labels = data
            if self.device is not None:
                # Move data to adequate device
                inputs, labels = map(lambda x: x.to(self.device,
                                                    non_blocking=self.pin_memory),
                                     (inputs, labels))
            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            assert torch.isnan(loss) == torch.tensor([0], dtype=torch.uint8).to(self.device)
            loss.backward()
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                probs = F.softmax(outputs, dim=1)
                self.train_accuracy.append(
                    calc_accuracy_torch(probs, labels, self.device).item())
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

            if self.checkpoint_steps > 0:
                if self.steps % self.checkpoint_steps == 0:
                    self._save_checkpoint(save_at_steps=True)

        return

    def test(self, time):
        """
        Single evaluation on the entire provided test dataset.
        Return accuracy, mean test loss, and an array of predicted probabilities
        """
        test_loss = 0.
        n_correct = 0  # Track the number of correct classifications

        # Set model in eval mode
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                # Get inputs
                inputs, labels = data
                if self.device is not None:
                    inputs, labels = map(lambda x: x.to(self.device),
                                         (inputs, labels))
                outputs = self.model(inputs)
                test_loss += self.test_criterion(outputs, labels).item()
                probs = F.softmax(outputs, dim=1)
                n_correct += torch.sum(torch.argmax(probs, dim=1) == labels).item()

        test_loss = test_loss / len(self.testloader)
        accuracy = n_correct / len(self.testloader.dataset)

        print(f"Test Loss: {np.round(test_loss, 3)}; "
              f"Test Accuracy: {np.round(100.0*accuracy, 1)}%; "
              f"Time Per Epoch: {np.round(time/60.0,1)} min")

        # Log statistics
        self.test_loss.append(test_loss)
        self.test_accuracy.append(accuracy)
        self.test_eval_steps.append(self.steps)
        return


def calc_accuracy_torch(y_probs, y_true, device=None, weights=None):
    if weights is None:
        if device is None:
            accuracy = torch.mean((torch.argmax(y_probs, dim=1) == y_true).to(dtype=torch.float64))
        else:
            accuracy = torch.mean(
                (torch.argmax(y_probs, dim=1) == y_true).to(device, torch.float64))
    else:
        if device is None:
            weights.to(dtype=torch.float64)
            accuracy = torch.mean(
                weights * (torch.argmax(y_probs, dim=1) == y_true).to(dtype=torch.float64))
        else:
            weights.to(device=device, dtype=torch.float64)
            accuracy = torch.mean(
                weights * (torch.argmax(y_probs, dim=1) == y_true).to(device=device,
                                                                      dtype=torch.float64))
    return accuracy
