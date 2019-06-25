import context
from prior_networks.training import Trainer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Dict, Any
from prior_networks.util_pytorch import calc_accuracy_torch


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
                 test_criterion=None):
        super().__init__(model, criterion, train_dataset, test_dataset,
                         optimizer, scheduler, optimizer_params,
                         scheduler_params, batch_size, device, log_interval,
                         test_criterion)

        assert len(train_dataset) == len(ood_dataset)
        self.oodloader = DataLoader(ood_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=1)

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
                inputs, labels, ood_inputs = map(lambda x: x.to(self.device),
                                                 (inputs, labels, ood_inputs))

            # zero the parameter gradients
            self.optimizer.zero_grad()

            inputs = torch.cat((inputs, ood_inputs), dim=0)
            outputs = self.model(inputs)
            id_outputs, ood_outputs = torch.chunk(outputs, 2, dim=0)
            loss = self.criterion((id_outputs, ood_outputs), (labels, None))

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
        return


# def train_procedure_with_ood(model, train_dataset, ood_dataset, test_dataset, n_epochs=80, lr=0.001, lr_decay=0.9,
#                              batch_size=50,
#                              match_dataset_length=False,
#                              id_criterion=None,
#                              ood_criterion=None,
#                              gamma=1.0,
#                              weight_decay=1e-8,
#                              print_progress='test',
#                              device=None):
#     print_train, print_test = _get_print_progress_vars(print_progress)
#
#     if id_criterion is None:
#         id_criterion = DirichletPriorNetLoss(target_concentration=100.0)
#     if ood_criterion is None:
#         ood_criterion = DirichletPriorNetLoss(target_concentration=0.0)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay, last_epoch=-1)
#     if match_dataset_length:
#         # todo: Add functionality if OOD and regular datasets not the same lengths
#         # ood_dataset = adjust_dataset_length
#         pass
#     assert len(train_dataset) == len(ood_dataset)
#     trainloader = DataLoader(train_dataset, batch_size=batch_size,
#                              shuffle=True, num_workers=1, drop_last=True)
#     oodloader = DataLoader(ood_dataset, batch_size=batch_size,
#                            shuffle=True, num_workers=1, drop_last=True)
#     testloader = DataLoader(test_dataset, batch_size=batch_size,
#                             shuffle=False, num_workers=1)
#
#     # Lists for storing training metrics
#     train_loss, train_accuracy = [], []
#     # Lists for storing test metrics
#     test_loss, test_accuracy, test_steps = [], [], []
#
#     for epoch in range(n_epochs):
#         if print_progress:
#             print(f"Epoch: {epoch}")
#         # Train
#         scheduler.step()
#         epoch_train_loss, epoch_train_accuracy = train_single_epoch_with_ood(model, trainloader, oodloader,
#                                                                              id_criterion, ood_criterion, gamma, optimizer,
#                                                                              print_progress=print_train,
#                                                                              device=device)
#         train_loss += epoch_train_loss
#         train_accuracy += epoch_train_accuracy
#         # Test
#         accuracy, loss = test(model, testloader, print_progress=print_test, device=device)
#         test_loss.append(loss)
#         test_accuracy.append(accuracy)
#         test_steps.append(len(train_loss))
#     return train_loss, train_accuracy, test_loss, test_accuracy, test_steps
