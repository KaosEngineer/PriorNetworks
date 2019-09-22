from typing import Dict, Any
import sys, os
import torch
import numpy as np
import torch.nn.functional as F
from collections import Counter
import math
import time

from torch.nn.utils import clip_grad_norm_
from prior_networks.training import Trainer, calc_accuracy_torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class _TempScheduler(object):
    def __init__(self, init_temp, last_epoch=-1):
        if last_epoch == -1:
            last_epoch = 0

        self.temp = init_temp
        self.step(last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.temp = self.update_temp()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_temp(self):
        raise NotImplementedError

    def update_temp(self):
        raise NotImplementedError


class MultiStepTempScheduler(_TempScheduler):
    def __init__(self, init_temp, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepTempScheduler, self).__init__(init_temp, last_epoch)
        self.temp = init_temp - 1.0

    def update_temp(self):
        if self.last_epoch not in self.milestones:
            return self.temp
        return self.temp * (self.gamma ** (self.milestones[self.last_epoch]))

    def get_temp(self):
        return self.temp + 1.0


class LRTempScheduler(_TempScheduler):
    def __init__(self, init_temp, decay_epoch, decay_length, min_temp=1.0, last_epoch=-1):
        assert decay_length > 0
        assert decay_epoch > 0
        self.decay_epoch = decay_epoch
        self.decay_length = decay_length
        self.init_temp = init_temp
        self.min_temp = min_temp
        super(LRTempScheduler, self).__init__(init_temp, last_epoch)

    def update_temp(self):
        if self.last_epoch <= self.decay_epoch:
            return self.temp
        elif self.last_epoch >= self.decay_epoch + self.decay_length:
            return self.min_temp
        else:
            slope = (self.init_temp - self.min_temp) / self.decay_length
            return self.init_temp - slope * (self.last_epoch - self.decay_epoch)

    def get_temp(self):
        return self.temp


class TrainerDistillation(Trainer):
    def __init__(self,
                 model,
                 criterion,
                 train_dataset,
                 test_dataset,
                 optimizer,
                 temp_scheduler,
                 scheduler=None,
                 optimizer_params: Dict[str, Any] = None,
                 scheduler_params: Dict[str, Any] = None,
                 temp_scheduler_params: Dict[str, Any] = None,
                 batch_size=50,
                 device=None,
                 log_interval: int = 100,
                 test_criterion=None,
                 clip_norm=10.0,
                 num_workers=4,
                 pin_memory=False,
                 checkpoint_path='./',
                 checkpoint_steps=0):
        super().__init__(model=model, criterion=criterion, train_dataset=train_dataset,
                         test_dataset=test_dataset, optimizer=optimizer, scheduler=scheduler,
                         optimizer_params=optimizer_params, scheduler_params=scheduler_params,
                         batch_size=batch_size, device=device, log_interval=log_interval,
                         test_criterion=test_criterion, num_workers=num_workers,
                         pin_memory=pin_memory,
                         clip_norm=clip_norm,
                         checkpoint_path=checkpoint_path, checkpoint_steps=checkpoint_steps)

        self.temp_scheduler = temp_scheduler(**temp_scheduler_params)

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
            'temp_scheduler_state_dict': self.temp_scheduler.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss
        }, os.path.join(self.checkpoint_path, checkpoint_name))

    def load_checkpoint(self,
                        checkpoint_path,
                        load_opt_state=False,
                        load_scheduler_state=False,
                        load_tscheduler_state=False,
                        map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.steps = checkpoint['steps']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_loss = checkpoint['train_loss']
        self.test_loss = checkpoint['test_loss']

        if load_opt_state:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_scheduler_state:
            self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if load_tscheduler_state:
            self.temp_scheduler.load_state_dict(checkpoint['temp_scheduler_state_dict'])

    def train(self, n_epochs=None, n_iter=None, resume=False):
        # Calc num of epochs
        init_epoch = 0
        if n_epochs is None:
            assert isinstance(n_iter, int)
            n_epochs = math.ceil(n_iter / len(self.trainloader))
        else:
            assert isinstance(n_epochs, int)

        if resume:
            init_epoch = math.floor(self.steps / len(self.trainloader))

        for epoch in range(init_epoch, n_epochs):
            print(f'Training epoch: {epoch + 1} / {n_epochs}')
            # Train
            start = time.time()
            self._train_single_epoch()
            self._save_checkpoint()
            # Test
            self.test(time=time.time() - start)
            self.scheduler.step()
            self.temp_scheduler.step()
        return

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        for i, data in enumerate(self.trainloader, 0):
            # Get inputs
            inputs, labels, logits = data
            if self.device is not None:
                # Move data to adequate device
                inputs, labels, logits = map(lambda x: x.to(self.device,
                                                            non_blocking=self.pin_memory),
                                             (inputs, labels, logits))

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            precision = torch.mean(torch.sum(torch.exp(outputs), dim=1))
            temp = self.temp_scheduler.get_temp()
            loss = self.criterion(outputs, logits, temp)

            assert torch.isnan(loss) == torch.tensor([0], dtype=torch.uint8).to(self.device)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                probs = F.softmax(outputs, dim=1)
                accuracy = calc_accuracy_torch(probs, labels, self.device).item()
                self.train_accuracy.append(accuracy)
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

            if self.checkpoint_steps > 0:
                if self.steps % self.checkpoint_steps == 0:
                    self._save_checkpoint(save_at_steps=True)
        with open('./LOG.txt', 'a') as f:
            f.write(f"Train Loss: {np.round(loss.item(), 3)}; "
                    f"Train Error: {np.round(100.0 * (1.0-accuracy), 1)}; "
                    f"Train Mean Precision: {np.round(precision.item(), 1)}; ")
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
                inputs, labels, logits = data
                if self.device is not None:
                    inputs, labels, logits = map(lambda x: x.to(self.device),
                                                 (inputs, labels, logits))
                outputs = self.model(inputs)
                temp = self.temp_scheduler.get_temp()
                loss = self.criterion(outputs, logits, temp).item()
                precision = torch.mean(torch.sum(torch.exp(outputs), dim=1)).item()

                test_loss += self.test_criterion(outputs, labels).item()
                probs = F.softmax(outputs, dim=1)
                n_correct += torch.sum(torch.argmax(probs, dim=1) == labels).item()

        test_loss = test_loss / len(self.testloader)
        accuracy = n_correct / len(self.testloader.dataset)

        print(f"Test Loss: {np.round(test_loss, 3)}; "
              f"Criterion Loss: {np.round(loss, 1)}; "
              f"Test Error: {np.round(100.0 * (1.0-accuracy), 1)}%; "
              f"Test Mean Precision: {np.round(precision, 1)}; "
              f"Time Per Epoch: {np.round(time / 60.0, 1)} min")

        with open('./LOG.txt', 'a') as f:
            f.write(f"Test Loss: {np.round(test_loss, 3)}; "
                    f"Criterion Loss: {np.round(loss, 1)}; "
                    f"Test Error: {np.round(100.0 * (1.0-accuracy), 1)}; "
                    f"Test Mean Precision: {np.round(precision, 1)}; "
                    f"Time Per Epoch: {np.round(time / 60.0, 1)} min.\n")

        # Log statistics
        self.test_loss.append(test_loss)
        self.test_accuracy.append(accuracy)
        self.test_eval_steps.append(self.steps)
        return
