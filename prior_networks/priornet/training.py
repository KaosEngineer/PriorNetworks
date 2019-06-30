from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prior_networks.training import Trainer, calc_accuracy_torch


class TrainerWithOOD(Trainer):
    def __init__(self, model, criterion,
                 train_dataset, ood_dataset, test_dataset,
                 optimizer,
                 scheduler=None,
                 optimizer_params: Dict[str, Any] = None,
                 scheduler_params: Dict[str, Any] = None,
                 batch_size=50,
                 device=None,
                 log_interval: int = 100,
                 test_criterion=None,
                 pin_memory=False,
                 checkpoint_path='./',
                 checkpoint_steps=0):
        super().__init__(model, criterion, train_dataset, test_dataset,
                         optimizer, scheduler, optimizer_params,
                         scheduler_params, batch_size, device, log_interval,
                         test_criterion, pin_memory, checkpoint_path, checkpoint_steps)

        assert len(train_dataset) == len(ood_dataset)
        self.oodloader = DataLoader(ood_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=1, pin_memory=self.pin_memory)

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        for i, (data, ood_data) in enumerate(
                zip(self.trainloader, self.oodloader), 0):
            # Get inputs
            inputs, labels = data
            ood_inputs, _ = ood_data
            if self.device is not None:
                # Move data to adequate device
                inputs, labels, ood_inputs = map(lambda x: x.to(self.device, non_blocking=self.pin_memory),
                                                 (inputs, labels, ood_inputs))

            # zero the parameter gradients
            self.optimizer.zero_grad()

            inputs = torch.cat((inputs, ood_inputs), dim=0)
            outputs = self.model(inputs)
            id_outputs, ood_outputs = torch.chunk(outputs, 2, dim=0)
            loss = self.criterion((id_outputs, ood_outputs), (labels, None))
            assert torch.isnan(loss) == torch.tensor([ 0 ], dtype=torch.uint8).to(self.device)
            loss.backward()
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                probs = F.softmax(id_outputs, dim=1)
                self.train_accuracy.append(
                    calc_accuracy_torch(probs, labels, self.device).item())
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

            if self.steps % self.checkpoint_steps == 0 and self.checkpoint_steps > 0:
                self._save_checkpoint(save_at_steps=True)
        return


class TrainerWithOODJoint(Trainer):
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
                 pin_memory=False,
                 checkpoint_path='./',
                 checkpoint_steps=0):
        super().__init__(model, criterion, train_dataset, test_dataset,
                         optimizer, scheduler, optimizer_params,
                         scheduler_params, batch_size, device, log_interval,
                         test_criterion, pin_memory, checkpoint_path=checkpoint_path, checkpoint_steps=checkpoint_steps)

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        for i,  data in enumerate(self.trainloader, 0):
            # Get inputs
            inputs, labels = data
            if self.device is not None:
                # Move data to adequate device
                inputs, *labels = map(lambda x: x.to(self.device, non_blocking=self.pin_memory),
                                    (inputs, *labels))

            # zero the parameter gradients
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, *labels)
            assert torch.isnan(loss) == torch.tensor([0], dtype=torch.uint8).to(self.device)
            loss.backward()
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                probs = F.softmax(outputs, dim=1)
                weights = labels[1]/torch.max(labels[1])
                self.train_accuracy.append(
                    calc_accuracy_torch(probs, labels[0], self.device, weights=weights).item())
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

            if self.steps % self.checkpoint_steps == 0 and self.checkpoint_steps > 0:
                self._save_checkpoint(save_at_steps=True)
        return
