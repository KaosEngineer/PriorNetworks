from typing import Dict, Any
import sys
import torch
import numpy as np
import torch.nn.functional as F
from collections import Counter

from prior_networks.training import Trainer, calc_accuracy_torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class MultiStepTempScheduler(object):
    def __init__(self, temp, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma

        if last_epoch == -1:
            last_epoch = 0

        self.step(last_epoch)
        self.base_temp = temp - 1

    def update_temp(self):
        if self.last_epoch not in self.milestones:
            return self.base_temp
        return self.base_temp * (self.gamma ** (self.milestones[self.last_epoch]))

    def get_temp(self):
        return self.temp + 1.0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.temp = self.update_temp()



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
                         checkpoint_path=checkpoint_path, checkpoint_steps=checkpoint_steps)

        self.temp_scheduler = temp_scheduler(**temp_scheduler_params)

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
            loss = self.criterion(outputs, logits)

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
                inputs, labels, logits = data
                if self.device is not None:
                    inputs, labels, logits = map(lambda x: x.to(self.device),
                                                 (inputs, labels, logits))
                outputs = self.model(inputs)
                test_loss += self.test_criterion(outputs, labels).item()
                probs = F.softmax(outputs, dim=1)
                n_correct += torch.sum(torch.argmax(probs, dim=1) == labels).item()

        test_loss = test_loss / len(self.testloader)
        accuracy = n_correct / len(self.testloader.dataset)

        print(f"Test Loss: {np.round(test_loss, 3)}; "
              f"Test Accuracy: {np.round(100.0 * accuracy, 1)}%; "
              f"Time Per Epoch: {np.round(time / 60.0, 1)} min")

        # Log statistics
        self.test_loss.append(test_loss)
        self.test_accuracy.append(accuracy)
        self.test_eval_steps.append(self.steps)
        return


