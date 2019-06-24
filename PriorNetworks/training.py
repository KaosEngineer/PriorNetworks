import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PriorNetworks.priornet.losses import DirichletPriorNetLoss
from PriorNetworks.util_pytorch import calc_accuracy_torch


class trainer(object):
    def __init__(self, optimizer, optimizer_params, lr_scheduler, lr_scheduler_params, model):
        self.optimier = optimizer
        self.optimier_params = optimizer_params
        self.lr_schedueler = lr_scheduler
        self.lr_schedueler_params = lr_scheduler_params
        self.model




# todo: put the training into a class to reduce the functional programming clutter
def test(model, testloader, batch_size=50, print_progress=True, device=None):
    """
    Single evaluation on the entire provided test dataset.
    Return accuracy, mean test loss, and an array of predicted probabilities
    """
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.
    n_correct = 0  # Track the number of correct classifications

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, labels = data
            if device is not None:
                inputs, labels = map(lambda x: x.to(device), (inputs, labels))
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            probs = F.softmax(outputs, dim=1)
            n_correct += torch.sum(torch.argmax(probs, dim=1) == labels).item()
    test_loss = test_loss / len(testloader)
    accuracy = n_correct / len(testloader.dataset)

    if print_progress:
        print(f"Test Loss: {test_loss}; Test Accuracy: {accuracy}")
    return accuracy, test_loss

def train_single_epoch_with_ood(model, trainloader, oodloader, id_criterion, ood_criterion, gamma, optimizer, print_progress=True, device=None):
    train_loss, train_accuracy = [], []

    model.train()
    running_loss = 0.0
    for i, (data, ood_data) in enumerate(zip(trainloader, oodloader), 0):
        # Get inputs
        inputs, labels = data
        ood_inputs, _ = ood_data
        if device is not None:
            inputs, labels, ood_inputs = map(lambda x: x.to(device), (inputs, labels, ood_inputs))


        # zero the parameter gradients
        optimizer.zero_grad()

        id_outputs = model(inputs)
        if gamma > 0.0:
            inputs = torch.cat((inputs, ood_inputs), dim=0)
            outputs = model(inputs)
            id_outputs, ood_outputs = torch.chunk(outputs, 2, dim=0)
            loss = id_criterion(id_outputs, labels) + gamma* ood_criterion(ood_outputs, None)
        else:
            loss = id_criterion(id_outputs, labels)
        loss.backward()
        optimizer.step()

        # log statistics
        train_loss.append(loss.item())
        probs = F.softmax(id_outputs, dim=1)
        train_accuracy.append(calc_accuracy_torch(probs, labels, device).item())

        # print statistics
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            if print_progress:
                print(f'[Step: {i + 1}] loss: {running_loss / 10.0}')
            running_loss = 0.0
    return train_loss, train_accuracy


def train_procedure_with_ood(model, train_dataset, ood_dataset, test_dataset, n_epochs=80, lr=0.001, lr_decay=0.9,
                             batch_size=50,
                             match_dataset_length=False,
                             id_criterion=None,
                             ood_criterion=None,
                             gamma=1.0,
                             weight_decay=1e-8,
                             print_progress='test',
                             device=None):
    print_train, print_test = _get_print_progress_vars(print_progress)

    if id_criterion is None:
        id_criterion = DirichletPriorNetLoss(target_concentration=100.0)
    if ood_criterion is None:
        ood_criterion = DirichletPriorNetLoss(target_concentration=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay, last_epoch=-1)
    if match_dataset_length:
        # todo: Add functionality if OOD and regular datasets not the same lengths
        # ood_dataset = adjust_dataset_length
        pass
    assert len(train_dataset) == len(ood_dataset)
    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=1, drop_last=True)
    oodloader = DataLoader(ood_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=1, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=1)

    # Lists for storing training metrics
    train_loss, train_accuracy = [], []
    # Lists for storing test metrics
    test_loss, test_accuracy, test_steps = [], [], []

    for epoch in range(n_epochs):
        if print_progress:
            print(f"Epoch: {epoch}")
        # Train
        scheduler.step()
        epoch_train_loss, epoch_train_accuracy = train_single_epoch_with_ood(model, trainloader, oodloader,
                                                                             id_criterion, ood_criterion, gamma, optimizer,
                                                                             print_progress=print_train,
                                                                             device=device)
        train_loss += epoch_train_loss
        train_accuracy += epoch_train_accuracy
        # Test
        accuracy, loss = test(model, testloader, print_progress=print_test, device=device)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
        test_steps.append(len(train_loss))
    return train_loss, train_accuracy, test_loss, test_accuracy, test_steps


def _get_print_progress_vars(print_progress):
    """Helper function for setting the right print variables"""
    if print_progress == 'all':
        print_train = True
        print_test = True
    elif print_progress == 'test':
        print_train = False
        print_test = True
    else:
        print_train = False
        print_test = False
    return print_train, print_test

    # # class Training(object):
    # def train_single_epoch(model, trainloader, criterion, optimizer, print_progress=True, device=None):
    #     train_loss, train_accuracy = [], []
    #
    #     model.train()
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # Get inputs
    #         inputs, labels = data
    #         if device is not None:
    #             # Move data to adequate device
    #             inputs, labels = map(lambda x: x.to(device), (inputs, labels))
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # log statistics
    #         train_loss.append(loss.item())
    #         probs = F.softmax(outputs, dim=1)
    #         train_accuracy.append(calc_accuracy_torch(probs, labels, device).item())
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if (i + 1) % 10 == 0:
    #             if print_progress:
    #                 print(f'[Step: {i + 1}] loss: {running_loss / 10.0}')
    #             running_loss = 0.0
    #     return train_loss, train_accuracy
    #
    # def train_procedure(model, train_dataset, test_dataset, n_epochs=None, n_iter=None, lr=0.001, lr_decay=0.9,
    #                     batch_size=50,
    #                     loss=nn.CrossEntropyLoss,
    #                     print_progress='all',
    #                     device=None):
    #
    #     # Get bool variables for what to print during training
    #     print_train, print_test = _get_print_progress_vars(print_progress)
    #
    #     criterion = loss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay, last_epoch=-1)
    #     trainloader = DataLoader(train_dataset, batch_size=batch_size,
    #                              shuffle=True, num_workers=1)
    #     testloader = DataLoader(test_dataset, batch_size=batch_size,
    #                             shuffle=False, num_workers=1)
    #
    #     # Lists for storing training metrics
    #     train_loss, train_accuracy = [], []
    #     # Lists for storing test metrics
    #     test_loss, test_accuracy, test_steps = [], [], []
    #
    #     # Calc num of epochs
    #     if n_epochs is None:
    #         n_epochs = math.ceil(n_iter / len(trainloader))
    #
    #     for epoch in range(n_epochs):
    #         if print_progress:
    #             print(f"Epoch: {epoch}")
    #         # Train
    #         scheduler.step()
    #         epoch_train_loss, epoch_train_accuracy = train_single_epoch(model, trainloader, criterion, optimizer,
    #                                                                     print_progress=print_train, device=device)
    #         train_loss += epoch_train_loss
    #         train_accuracy += epoch_train_accuracy
    #         # Test
    #         accuracy, loss = test(model, testloader, print_progress=print_test, device=device)
    #         test_loss.append(loss)
    #         test_accuracy.append(accuracy)
    #         test_steps.append(len(train_loss))
    #     return train_loss, train_accuracy, test_loss, test_accuracy, test_steps

        # def train_single_epoch_endd(model, teacher_ensemble, trainloader, criterion, optimizer, print_progress=True,
#                             device=None):
#     train_loss, train_accuracy = [], []
#
#     model.train()
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # Get inputs
#         inputs, labels = data
#         if device is not None:
#             inputs, labels = map(lambda x: x.to(device), (inputs, labels))
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         model_logits = model(inputs)
#         ensemble_logits = teacher_ensemble(inputs)
#         loss = criterion(model_logits, ensemble_logits)
#         loss.backward()
#         optimizer.step()
#
#         # log statistics
#         train_loss.append(loss.item())
#         probs = F.softmax(model_logits, dim=1)
#         train_accuracy.append(calc_accuracy_torch(probs, labels, device).item())
#
#         # print statistics
#         running_loss += loss.item()
#         if (i + 1) % 10 == 0:
#             if print_progress:
#                 print(f'[Step: {i + 1}] loss: {running_loss / 10.0}')
#             running_loss = 0.0
#     return train_loss, train_accuracy
#
#
# def train_procedure_endd(model, teacher_ensemble, train_dataset, test_dataset, n_epochs=80, lr=0.001, lr_decay=0.9,
#                          batch_size=50,
#                          criterion=None,
#                          print_progress='test', device=None):
#     print_train, print_test = _get_print_progress_vars(print_progress)
#
#     if criterion is None:
#         criterion = EnDDSamplesLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay, last_epoch=-1)
#
#     trainloader = DataLoader(train_dataset, batch_size=batch_size,
#                              shuffle=True, num_workers=1, drop_last=True)
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
#         epoch_train_loss, epoch_train_accuracy = train_single_epoch_endd(model, teacher_ensemble, trainloader,
#                                                                          criterion, optimizer,
#                                                                          print_progress=print_train, device=device)
#         train_loss += epoch_train_loss
#         train_accuracy += epoch_train_accuracy
#         # Test
#         accuracy, loss = test(model, testloader, print_progress=print_test, device=device)
#         test_loss.append(loss)
#         test_accuracy.append(accuracy)
#         test_steps.append(len(train_loss))
#     return train_loss, train_accuracy, test_loss, test_accuracy, test_steps
#